"""LLM layer orchestration (plan P4 steps 3-7, D4: run every LLM stage once, simulate everything offline).

For a set of function ids:  Pass 1 -> retrieval (strongest hypothesis) -> Pass 2 neutral AND refute
(+ zero-shot on test). Every call goes through the hash cache; `llm/index.parquet` records, per (id, stage),
the cache key, parsed output, hypothesis used, demos and token/cost usage (input to P5-P7).
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd

from ..common import atomic_to_parquet
from ..config import dataset_dir, get_logger, load_config
from ..cpg.joern import load_facts
from ..data.io import load_records
from ..features.store import load_store
from ..verify.cpg_decide import guard_candidates
from .client import LLMClient, run_requests
from .context import build_context, slice_text_for_line
from .prompts import current_hashes, prompt_hash, render, schema
from .retrieval import Retriever, get_text_encoder

log = get_logger("p4.runner")
INDEX_COLUMNS = ["id", "stage", "key", "status", "parsed", "hyp", "demos", "model", "provider", "prompt_tokens",
                 "completion_tokens", "reasoning_tokens", "cost_usd", "latency_s"]


def index_path(dataset: str):
    from .client import mock_mode

    return dataset_dir(dataset, "llm") / ("index.mock.parquet" if mock_mode() else "index.parquet")


def load_index(dataset: str) -> pd.DataFrame:
    path = index_path(dataset)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=INDEX_COLUMNS)


def save_index_rows(dataset: str, rows: list[dict], replace: tuple[list[str], list[str]] | None = None) -> None:
    """Upsert rows keyed by (id, stage). `replace=(ids, stages)` first drops every existing row of those ids/stages,
    so a re-run leaves no stale row behind (e.g. Pass 2 of a function whose new Pass 1 has no hypothesis)."""
    old = load_index(dataset)
    if replace is not None:
        ids, stages = replace
        old = old[~(old["id"].isin(set(ids)) & old["stage"].isin(set(stages)))]
    if not rows and replace is None:
        return
    new = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(["id", "stage"], keep="last")
    atomic_to_parquet(merged, index_path(dataset))


def _row(entry: dict, stage: str, hyp=None, demos=None) -> dict:
    usage = entry.get("usage") or {}
    return {"id": entry.get("id"), "stage": stage, "key": entry.get("key"), "status": entry.get("status"),
            "parsed": json.dumps(entry.get("parsed")) if entry.get("parsed") is not None else None,
            "hyp": json.dumps(hyp) if hyp is not None else None,
            "demos": json.dumps([d["demo_id"] for d in demos]) if demos else None,
            "model": entry.get("model"), "provider": entry.get("provider"),
            "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0), "cost_usd": entry.get("cost_usd", 0.0),
            "latency_s": entry.get("latency_s", 0.0)}


class FrozenPromptError(RuntimeError):
    pass


def check_frozen(stages: list[str]) -> None:
    frozen = load_config("llm").get("prompt_hashes") or {}
    current = current_hashes()
    bad = [s for s in stages if s in current and frozen.get(s) != current[s]]
    if bad:
        raise FrozenPromptError(
            f"prompts not frozen or changed since freezing: {bad}. Run `run_p4_llm freeze` after the dev-set loop (D6).")


class LLMLayer:
    def __init__(self, dataset: str, model: str | None = None, provider: dict | None = None,
                 max_tokens: int | None = None, response_format: str | None = None) -> None:
        self.dataset = dataset
        self.cfg = load_config("llm")
        self.sink_apis = set(load_config("cpg")["sink_apis"])
        self.records = load_records(dataset, ["id", "code", "label", "subset", "split", "dev_set", "pair_id", "pair_role",
                                              "code_hash", "cwe", "in_main"]).set_index("id", drop=False)
        self.client = LLMClient(dataset, self.cfg, model=model, provider=provider, max_tokens=max_tokens,
                                response_format=response_format)
        self._facts: dict[str, dict] = {}
        self._contexts: dict[str, dict] = {}
        self._retriever: Retriever | None = None
        self._encoder = None

    # ---------------------------------------------------------------- inputs
    def facts(self, ids: list[str]) -> dict[str, dict]:
        missing = set(ids) - set(self._facts)
        if missing:
            self._facts.update(load_facts(self.dataset, missing))
        return self._facts

    def context(self, record_id: str) -> dict:
        if record_id not in self._contexts:
            row = self.records.loc[record_id]
            self._contexts[record_id] = build_context(record_id, row["code"], self._facts.get(record_id),
                                                      self.cfg["prompt"], self.sink_apis)
        return self._contexts[record_id]

    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.dataset)
        return self._retriever

    def _run(self, jobs: list[dict]) -> list[dict]:
        if not jobs:
            return []
        return asyncio.run(run_requests(self.client, jobs, int(self.cfg.get("concurrency", 12))))

    # ---------------------------------------------------------------- stages
    def pass1(self, ids: list[str], stage: str = "pass1", extra_params: dict | None = None) -> dict[str, dict]:
        self.facts(ids)
        jobs = []
        for record_id in ids:
            ctx = self.context(record_id)
            jobs.append({"record_id": record_id, "stage": stage, "prompt_hash": prompt_hash("pass1"),
                         "messages": render("pass1", ctx=ctx), "schema": schema("pass1"),
                         "extra_params": extra_params, "mock_hint": {"sinks": ctx["sinks"]}})
        results = {entry["id"]: entry for entry in self._run(jobs)}
        save_index_rows(self.dataset, [_row(e, stage) for e in results.values()])
        return results

    def _queries(self, ids: list[str], pass1: dict[str, dict], use_hypothesis: bool) -> dict[str, dict]:
        """Strongest hypothesis + slice text + query embedding for each id."""
        queries: dict[str, dict] = {}
        for record_id in ids:
            entry = pass1.get(record_id)
            hyps = ((entry or {}).get("parsed") or {}).get("hypotheses") or []
            if use_hypothesis and not hyps:
                continue
            hyp = hyps[0] if (use_hypothesis and hyps) else None
            ctx = self.context(record_id)
            code = self.records.loc[record_id, "code"]
            if hyp:
                line = int(hyp.get("sink_line") or 0)
            else:
                line = ctx["sinks"][0]["line"] if ctx["sinks"] else None
            queries[record_id] = {"hyp": hyp, "line": line, "slice_text": slice_text_for_line(ctx, code, line),
                                  "sink": next((s["name"] for s in ctx["all_sinks"] if line and abs(s["line"] - line) <= 1), None)}
        if not queries:
            return queries
        retriever = self.retriever()
        if retriever.has_pairs:
            if self._encoder is None:
                self._encoder = get_text_encoder()
            emb = self._encoder.encode([q["slice_text"] for q in queries.values()], progress_every=2000)
        else:
            store = load_store(self.dataset)
            row_of = {rid: i for i, rid in enumerate(store["ids"]["id"])}
            emb = np.asarray(store["unixcoder"][[row_of[i] for i in queries]], dtype=np.float32)
        for (record_id, query), vector in zip(queries.items(), emb):
            query["emb"] = vector
        return queries

    def pass2(self, ids: list[str], pass1: dict[str, dict], modes=("neutral", "refute"), prefix: str = "pass2",
              use_hypothesis: bool = True, single_demo: bool = False, extra_params: dict | None = None) -> dict:
        queries = self._queries(ids, pass1, use_hypothesis)
        retriever = self.retriever()
        jobs, meta = [], {}
        for record_id, query in queries.items():
            row = self.records.loc[record_id]
            hyp = query["hyp"]
            demos, _ = retriever.query(query["emb"], cwe=(hyp or {}).get("cwe"), sink=query["sink"],
                                       exclude_hashes={row["code_hash"]}, exclude_pair=row["pair_id"],
                                       single_demo=single_demo)
            ctx = self.context(record_id)
            hint = {"sink_line": query["line"], "sink": query["sink"],
                    "guard_candidates": guard_candidates(self._facts.get(record_id), query["line"])}
            for mode in modes:
                base = f"pass2_{mode}"
                stage = base if prefix == "pass2" else f"{prefix}_{mode}"
                jobs.append({"record_id": record_id, "stage": stage, "prompt_hash": prompt_hash(base),
                             "messages": render(base, ctx=ctx, hyp=hyp, demos=demos, slice_text=query["slice_text"]),
                             "schema": schema(base), "extra_params": extra_params, "mock_hint": hint})
                meta[(record_id, stage)] = (hyp, demos)
        results = self._run(jobs)
        rows, by_stage = [], {}
        for entry in results:
            hyp, demos = meta[(entry["id"], entry["stage"])]
            rows.append(_row(entry, entry["stage"], hyp, demos))
            by_stage.setdefault(entry["stage"], {})[entry["id"]] = entry
        stages = [f"pass2_{m}" if prefix == "pass2" else f"{prefix}_{m}" for m in modes]
        save_index_rows(self.dataset, rows, replace=(list(ids), stages))
        return by_stage

    def zeroshot(self, ids: list[str], stage: str = "zeroshot") -> dict[str, dict]:
        self.facts(ids)
        jobs = [{"record_id": i, "stage": stage, "prompt_hash": prompt_hash("zeroshot"),
                 "messages": render("zeroshot", ctx=self.context(i)), "schema": schema("zeroshot")} for i in ids]
        results = {entry["id"]: entry for entry in self._run(jobs)}
        save_index_rows(self.dataset, [_row(e, stage) for e in results.values()])
        return results

    def max_similarity(self, ids: list[str]) -> pd.DataFrame:
        """Router feature max_sim_to_demo_bank for every LLM target (function embedding vs bank)."""
        store = load_store(self.dataset)
        row_of = {rid: i for i, rid in enumerate(store["ids"]["id"])}
        emb = np.asarray(store["unixcoder"][[row_of[i] for i in ids]], dtype=np.float32)
        sims = self.retriever().max_similarity(emb)
        frame = pd.DataFrame({"id": ids, "max_sim_to_demo_bank": sims})
        frame.to_parquet(dataset_dir(self.dataset, "retrieval") / "max_sim.parquet", index=False)
        return frame

    # ---------------------------------------------------------------- pipelines
    def run_main(self, ids: list[str], with_zeroshot: bool) -> dict:
        log.info("[%s] Pass 1 on %d functions", self.dataset, len(ids))
        p1 = self.pass1(ids)
        n_hyp = sum(1 for e in p1.values() if (e.get("parsed") or {}).get("hypotheses"))
        log.info("[%s] Pass 1 done: %d with hypotheses, %d invalid", self.dataset, n_hyp,
                 sum(1 for e in p1.values() if e.get("parsed") is None))
        p2 = self.pass2(ids, p1)
        log.info("[%s] Pass 2 done: %s", self.dataset, {k: len(v) for k, v in p2.items()})
        if with_zeroshot:
            self.zeroshot(ids)
        return {"pass1": len(p1), **{k: len(v) for k, v in p2.items()}, "spend_session_usd": self.client.spent_session}

    def pass1_results(self, ids: list[str], stage: str = "pass1") -> dict[str, dict]:
        """Pass-1 outputs from the cache (re-issued requests are cache hits, no API cost)."""
        return self.pass1(ids, stage)
