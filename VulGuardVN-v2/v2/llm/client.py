"""OpenRouter client with hash cache (plan P4 step 1, §6.6).

- Configuration D2: fixed provider (allow_fallbacks=false, require_parameters=true), reasoning.effort=low,
  temperature=0, seed=42, max_tokens=4096, JSON-schema output.
- Cache: append-only `artifacts/{dataset}/llm/llm_cache.jsonl`, key = sha256(model|provider|prompt_hash|messages|params).
  A key already in the cache is never re-requested (resume). Failed calls are not cached (retried next run).
- Retry with exponential backoff, 8-16 concurrent requests, token/cost/provider/latency logging, spend cap.
- Mock mode (VG2_LLM_MOCK=1 or --mock): deterministic offline answers for smoke tests, separate cache file.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import httpx
import jsonschema

from ..config import artifacts_dir, dataset_dir, get_logger, load_config

log = get_logger("p4.client")


def mock_mode() -> bool:
    return os.getenv("VG2_LLM_MOCK", "").strip() in {"1", "true", "yes"}


def cache_path(dataset: str) -> Path:
    name = "llm_cache.mock.jsonl" if mock_mode() else "llm_cache.jsonl"
    return dataset_dir(dataset, "llm") / name


def cache_key(model: str, provider: dict, prompt_hash: str, messages: list[dict], params: dict) -> str:
    blob = "|".join([model, json.dumps(provider, sort_keys=True), prompt_hash,
                     json.dumps(messages, sort_keys=True, ensure_ascii=False), json.dumps(params, sort_keys=True)])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_json(content: str | None, schema: dict) -> tuple[dict | None, str | None]:
    if not content:
        return None, "empty"
    text = content.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            jsonschema.validate(payload, schema)
            return payload, None
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:200]
    return None, error


class LLMCache:
    """In-memory view of one dataset's cache file, with append-only persistence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        self.lock = threading.Lock()
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                        self.entries[entry["key"]] = entry
                    except Exception:
                        continue

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, entry: dict) -> None:
        with self.lock:
            self.entries[entry["key"]] = entry
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def spend(self) -> float:
        return float(sum(e.get("cost_usd") or 0.0 for e in self.entries.values()))


def total_spend() -> float:
    """Spend over every dataset cache (real calls only) - compared with spend_cap_usd."""
    total = 0.0
    for path in artifacts_dir().glob("*/llm/llm_cache.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    total += float(json.loads(line).get("cost_usd") or 0.0)
                except Exception:
                    continue
    return total


class SpendCapReached(RuntimeError):
    pass


class LLMClient:
    def __init__(self, dataset: str, cfg: dict | None = None, model: str | None = None, provider: dict | None = None) -> None:
        self.cfg = cfg or load_config("llm")
        self.dataset = dataset
        self.model = model or self.cfg["model"]
        self.provider = provider if provider is not None else dict(self.cfg["provider"])
        self.mock = mock_mode()
        if self.mock:
            self.model = "mock/" + self.model
        self.cache = LLMCache(cache_path(dataset))
        self.price = self.cfg["price_snapshot"]
        self.cap = float(self.cfg.get("spend_cap_usd", 95.0))
        self.spent_before = 0.0 if self.mock else total_spend()
        self.spent_session = 0.0
        self.calls = 0
        self.failures = 0
        api_key_env = self.cfg.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key = (os.getenv(api_key_env) or "").strip().strip('"').strip("'")
        if not self.mock and not self.api_key:
            raise RuntimeError(f"{api_key_env} is empty - fill it in VulGuardVN-v2/.env (or use --mock)")

    def params(self, extra: dict | None = None) -> dict:
        params = {"temperature": self.cfg["temperature"], "seed": self.cfg["seed"], "max_tokens": self.cfg["max_tokens"],
                  "reasoning": self.cfg["reasoning"]}
        params.update(extra or {})
        return params

    def key(self, prompt_hash: str, messages: list[dict], params: dict) -> str:
        return cache_key(self.model, self.provider, prompt_hash, messages, params)

    async def complete(self, http: httpx.AsyncClient | None, semaphore: asyncio.Semaphore, *, record_id: str, stage: str,
                       prompt_hash: str, messages: list[dict], schema: dict, extra_params: dict | None = None,
                       mock_hint: dict | None = None) -> dict:
        params = self.params(extra_params)
        key = self.key(prompt_hash, messages, params)
        cached = self.cache.get(key)
        if cached is not None:
            # identical prompts (e.g. two windowed long functions) share one cached call: relabel for this record
            return {**cached, "id": record_id, "stage": stage}
        async with semaphore:
            if not self.mock and self.spent_before + self.spent_session >= self.cap:
                raise SpendCapReached(f"spend cap {self.cap} USD reached")
            started = time.perf_counter()
            if self.mock:
                response = _mock_response(stage, messages, mock_hint or {}, self.price)
            else:
                response = await self._request(http, messages, schema, params)
            latency = time.perf_counter() - started
        if response is None:
            self.failures += 1
            return {"key": key, "id": record_id, "stage": stage, "status": "error", "parsed": None}
        content = response.get("content")
        parsed, error = parse_json(content, schema)
        entry = {
            "key": key, "dataset": self.dataset, "id": record_id, "stage": stage, "model": self.model,
            "provider": response.get("provider"), "params": params, "prompt_hash": prompt_hash,
            "content": content, "finish_reason": response.get("finish_reason"), "usage": response.get("usage", {}),
            "cost_usd": response.get("cost_usd", 0.0), "latency_s": round(latency, 3),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "status": "ok" if parsed is not None else "invalid_json", "parse_error": error, "parsed": parsed,
        }
        self.cache.put(entry)
        self.calls += 1
        self.spent_session += float(entry["cost_usd"] or 0.0)
        return entry

    async def _request(self, http: httpx.AsyncClient, messages: list[dict], schema: dict, params: dict) -> dict | None:
        body = {
            "model": self.model, "messages": messages, "provider": self.provider,
            "temperature": params["temperature"], "seed": params["seed"], "max_tokens": params["max_tokens"],
            "reasoning": params["reasoning"], "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        }
        # params["repeat"] (P4 step 7) only changes the cache key; the request itself is identical
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "X-Title": "VulGuard v2 research"}
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        retries = int(self.cfg.get("max_retries", 6))
        for attempt in range(retries + 1):
            try:
                reply = await http.post(url, json=body, headers=headers, timeout=float(self.cfg.get("timeout_s", 180)))
                if reply.status_code == 200:
                    payload = reply.json()
                    if "error" in payload:
                        raise httpx.HTTPStatusError(str(payload["error"])[:200], request=reply.request, response=reply)
                    choice = (payload.get("choices") or [{}])[0]
                    usage = payload.get("usage") or {}
                    details = usage.get("completion_tokens_details") or {}
                    return {
                        "content": (choice.get("message") or {}).get("content"),
                        "finish_reason": choice.get("finish_reason"),
                        "provider": payload.get("provider"),
                        "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                                  "completion_tokens": usage.get("completion_tokens", 0),
                                  "reasoning_tokens": details.get("reasoning_tokens", 0)},
                        "cost_usd": float(usage.get("cost") or self._price(usage)),
                    }
                if reply.status_code in {400, 401, 402, 403}:
                    log.info("non-retryable HTTP %d: %s", reply.status_code, reply.text[:300])
                    if reply.status_code in {401, 402}:
                        raise SpendCapReached(f"HTTP {reply.status_code}: {reply.text[:200]}")
                    return None
                log.info("HTTP %d (attempt %d): %s", reply.status_code, attempt + 1, reply.text[:160])
            except SpendCapReached:
                raise
            except Exception as exc:  # noqa: BLE001 - network errors, timeouts, provider errors
                log.info("request error (attempt %d): %s", attempt + 1, str(exc)[:160])
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
        return None

    def _price(self, usage: dict) -> float:
        return (usage.get("prompt_tokens", 0) * float(self.price["input_per_m"]) +
                usage.get("completion_tokens", 0) * float(self.price["output_per_m"])) / 1e6


# ---------------------------------------------------------------- mock LLM (offline smoke tests)
def _mock_response(stage: str, messages: list[dict], hint: dict, price: dict) -> dict:
    text = messages[-1]["content"]
    digest = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(digest)
    if stage.endswith("pass1"):
        sinks = hint.get("sinks") or []
        hyps = []
        for sink in sinks[: rng.randint(0, 3)]:
            var = (sink.get("vars") or ["len"])[0]
            hyps.append({"cwe": rng.choice(["CWE-787", "CWE-125", "CWE-476", "CWE-190"]), "sink_line": int(sink["line"]),
                         "variable": var, "required_guard": f"{var} is within bounds", "rationale": "mock hypothesis"})
        content = json.dumps({"hypotheses": hyps})
    elif "zeroshot" in stage:
        content = json.dumps({"verdict": rng.choice(["vulnerable", "not_vulnerable"]), "evidence_lines": [1],
                              "confidence": round(rng.random(), 3)})
    else:
        guard = hint.get("guard_candidates") or []
        vulnerable = rng.random() < 0.5
        content = json.dumps({"verdict": "vulnerable" if vulnerable else "not_vulnerable",
                              "evidence_lines": [int(hint.get("sink_line") or 1)], "sink": str(hint.get("sink") or "?"),
                              "guard_line": None if vulnerable or not guard else int(rng.choice(guard)),
                              "confidence": round(rng.random(), 3)})
    usage = {"prompt_tokens": len(text) // 4 + 80, "completion_tokens": 260, "reasoning_tokens": 180}
    cost = (usage["prompt_tokens"] * float(price["input_per_m"]) + usage["completion_tokens"] * float(price["output_per_m"])) / 1e6
    return {"content": content, "finish_reason": "stop", "provider": "mock", "usage": usage, "cost_usd": cost}


async def run_requests(client: LLMClient, jobs: list[dict], concurrency: int | None = None, progress_every: int = 200) -> list[dict]:
    """jobs: dicts with record_id, stage, prompt_hash, messages, schema, extra_params, mock_hint."""
    semaphore = asyncio.Semaphore(int(concurrency or client.cfg.get("concurrency", 12)))
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
    results: list[dict | None] = [None] * len(jobs)
    started = time.perf_counter()
    done = 0
    async with httpx.AsyncClient(limits=limits, http2=False) as http:
        async def one(index: int, job: dict) -> None:
            nonlocal done
            results[index] = await client.complete(http, semaphore, **job)
            done += 1
            if progress_every and done % progress_every == 0:
                log.info("[%s] %d/%d calls | new=%d failures=%d | session spend %.3f USD | %.1f calls/s", client.dataset,
                         done, len(jobs), client.calls, client.failures, client.spent_session,
                         done / max(time.perf_counter() - started, 1e-6))
        tasks = [asyncio.create_task(one(i, job)) for i, job in enumerate(jobs)]
        try:
            await asyncio.gather(*tasks)
        except SpendCapReached:
            for task in tasks:
                task.cancel()
            raise
    return [r for r in results if r is not None]
