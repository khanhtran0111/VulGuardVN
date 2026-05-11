import json
import os
import re
import time
from pathlib import Path
from typing import Any

from common import CACHE_DIR, SHARED_MODELS_DIR, build_structure_summary, ensure_dir, load_json, stable_hash, truncate_text


EVIDENCE_SCHEMA_ENABLED = os.getenv("GRACE_EVIDENCE_AWARE_VERIFIER", "0").strip().lower() in {"1", "true", "yes", "on"}
CACHE_SCHEMA_VERSION = 2 if EVIDENCE_SCHEMA_ENABLED else 1
EXPECTED_RESPONSE_KEYS = {"label", "confidence", "reason"}
DEFAULT_MODEL_REPO_ID = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
DEFAULT_REASON_WORD_LIMIT = 48
DEFAULT_PROMPT_CODE_CHAR_LIMIT = int(os.getenv("GRACE_PROMPT_CODE_CHAR_LIMIT", "1800"))
DEFAULT_PROMPT_TOP_LINES_LIMIT = int(os.getenv("GRACE_PROMPT_TOP_LINES_LIMIT", "3"))
DEFAULT_PROMPT_TOP_LINE_CHAR_LIMIT = int(os.getenv("GRACE_PROMPT_TOP_LINE_CHAR_LIMIT", "120"))
DEFAULT_PROMPT_SLICES_CHAR_LIMIT = int(os.getenv("GRACE_PROMPT_SLICES_CHAR_LIMIT", "900"))
DEFAULT_PROMPT_NODE_INFO_CHAR_LIMIT = int(os.getenv("GRACE_PROMPT_NODE_INFO_CHAR_LIMIT", "900"))
DEFAULT_PROMPT_EDGE_INFO_CHAR_LIMIT = int(os.getenv("GRACE_PROMPT_EDGE_INFO_CHAR_LIMIT", "900"))
if EVIDENCE_SCHEMA_ENABLED:
    CHAT_SYSTEM_INSTRUCTION = (
        "You are a software vulnerability classifier. "
        "Do not output chain-of-thought, step-by-step reasoning, markdown, or any preamble. "
        'Your entire visible answer must be exactly one line that starts with FINAL_JSON: '
        "followed by a JSON object with keys label, confidence, reason, cwe_family, vulnerable_lines, sink_or_api, missing_guard."
    )
else:
    CHAT_SYSTEM_INSTRUCTION = (
        "You are a software vulnerability classifier. "
        "Do not output chain-of-thought, step-by-step reasoning, markdown, or any preamble. "
        'Your entire visible answer must be exactly one line that starts with FINAL_JSON: '
        "followed by a JSON object with keys label, confidence, reason."
    )


def default_local_model_dir(repo_id: str = DEFAULT_MODEL_REPO_ID) -> Path:
    return SHARED_MODELS_DIR / "local_llm" / repo_id.replace("/", "--")


def resolve_hf_token() -> str | None:
    for env_name in ["HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"]:
        value = os.getenv(env_name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


def is_model_downloaded(model_dir: Path) -> bool:
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))
    return (model_dir / "config.json").exists() and has_weights


def download_model_snapshot(repo_id: str, local_dir: Path | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `huggingface_hub`. Install it before downloading the local model."
        ) from exc

    target_dir = Path(local_dir or default_local_model_dir(repo_id))
    ensure_dir(target_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        token=resolve_hf_token(),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "pytorch_model*.bin",
            "*.txt",
            "*.model",
            "tokenizer*",
            "merges.txt",
            "vocab.*",
        ],
    )
    return target_dir


class LocalVulnLLMClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_REPO_ID,
        model_dir: Path | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
        cache_path: Path | None = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        auto_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir or default_local_model_dir(model_name))
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.auto_download = auto_download
        self.cache_path = cache_path or (CACHE_DIR / "local_vulnllm_cache.json")
        ensure_dir(self.cache_path.parent)
        self.cache = load_json(self.cache_path, default={})
        self._runtime: dict[str, Any] | None = None
        self._model = None
        self._tokenizer = None

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cache_key(self, prompt: str) -> str:
        return stable_hash(
            "\n".join(
                [
                    self.model_name,
                    str(self.model_dir),
                    f"load_in_4bit={self.load_in_4bit}",
                    f"max_new_tokens={self.max_new_tokens}",
                    f"temperature={self.temperature}",
                    CHAT_SYSTEM_INSTRUCTION,
                    prompt,
                ]
            )
        )

    def prompt_hash(self, prompt: str) -> str:
        return self._cache_key(prompt)

    def _invalidate_cache_key(self, cache_key: str) -> None:
        if cache_key in self.cache:
            self.cache.pop(cache_key, None)
            self._save_cache()

    def _get_cached_entry(self, prompt: str) -> dict | None:
        cache_key = self._cache_key(prompt)
        entry = self.cache.get(cache_key)
        if entry is None:
            return None
        try:
            normalized = _normalize_cache_entry(entry)
        except Exception:
            self._invalidate_cache_key(cache_key)
            return None
        if normalized != entry:
            self.cache[cache_key] = normalized
            self._save_cache()
        return normalized

    def is_cached(self, prompt: str) -> bool:
        return self._get_cached_entry(prompt) is not None

    def prepare(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "model_name": self.model_name,
            "model_dir": str(self.model_dir),
            "load_in_4bit": self.load_in_4bit,
            "device": str(self._model_input_device()),
        }

    def classify(self, prompt: str) -> dict:
        cached_entry = self._get_cached_entry(prompt)
        if cached_entry is not None:
            result = dict(cached_entry["parsed"])
            result["raw_text"] = cached_entry["raw_text"]
            result["cached"] = True
            result["finish_reason"] = cached_entry.get("finish_reason")
            result["usage_metadata"] = cached_entry.get("usage_metadata", {})
            result["device"] = cached_entry.get("device")
            return result

        cache_key = self._cache_key(prompt)
        response_payload = None
        try:
            response_payload = self._generate_response_payload(prompt)
            parsed = _parse_detection_payload(response_payload.get("raw_text", ""))
            cache_entry = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "model_name": self.model_name,
                "raw_text": response_payload.get("raw_text", ""),
                "parsed": parsed,
                "finish_reason": response_payload.get("finish_reason"),
                "usage_metadata": response_payload.get("usage_metadata", {}),
                "device": response_payload.get("device"),
                "cached_at_unix": time.time(),
            }
            self.cache[cache_key] = cache_entry
            self._save_cache()
            result = dict(parsed)
            result["raw_text"] = cache_entry["raw_text"]
            result["cached"] = False
            result["finish_reason"] = cache_entry["finish_reason"]
            result["usage_metadata"] = cache_entry["usage_metadata"]
            result["device"] = cache_entry["device"]
            return result
        except Exception as exc:
            self._invalidate_cache_key(cache_key)
            if response_payload is not None:
                finish_reason = response_payload.get("finish_reason")
                raw_preview = (response_payload.get("raw_text", "") or "")[:200]
                raise RuntimeError(
                    f"{exc} | finish_reason={finish_reason} | raw_text_preview={raw_preview!r}"
                ) from exc
            raise

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self.auto_download and not is_model_downloaded(self.model_dir):
            download_model_snapshot(self.model_name, self.model_dir)
        if not is_model_downloaded(self.model_dir):
            raise FileNotFoundError(
                f"Local model directory is missing or incomplete: {self.model_dir}. "
                "Run `python GRACE-improve/baseline/baseline2/00_verify_assets.py` first."
            )

        runtime = self._load_runtime()
        torch = runtime["torch"]
        AutoModelForCausalLM = runtime["AutoModelForCausalLM"]
        AutoTokenizer = runtime["AutoTokenizer"]
        BitsAndBytesConfig = runtime["BitsAndBytesConfig"]

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        config_payload = {}
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            try:
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config_payload = {}
        has_prequantized_config = bool(config_payload.get("quantization_config")) or "bnb-4bit" in self.model_name.lower()

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if self.load_in_4bit and not has_prequantized_config:
            model_kwargs["dtype"] = torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(self.model_dir, **model_kwargs)
        model.eval()

        self._runtime = runtime
        self._tokenizer = tokenizer
        self._model = model

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "Missing local LLM dependencies. Install `torch`, `transformers`, `accelerate`, "
                "`bitsandbytes`, and `huggingface_hub` before hybrid inference."
            ) from exc
        self._runtime = {
            "torch": torch,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "BitsAndBytesConfig": BitsAndBytesConfig,
        }
        return self._runtime

    def _model_input_device(self):
        if hasattr(self._model, "device") and self._model.device is not None:
            return self._model.device
        for parameter in self._model.parameters():
            return parameter.device
        raise RuntimeError("Could not determine local model device.")

    def _generate_response_payload(self, prompt: str) -> dict:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._runtime["torch"]

        messages = [
            {
                "role": "system",
                "content": CHAT_SYSTEM_INSTRUCTION,
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt

        model_inputs = tokenizer([rendered], return_tensors="pt")
        input_device = self._model_input_device()
        model_inputs = {name: tensor.to(input_device) for name, tensor in model_inputs.items()}
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_kwargs)

        new_tokens = generated[:, model_inputs["input_ids"].shape[-1] :]
        raw_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        generated_tokens = int(new_tokens.shape[-1])
        finish_reason = "length" if generated_tokens >= self.max_new_tokens else "stop"
        return {
            "raw_text": raw_text,
            "finish_reason": finish_reason,
            "usage_metadata": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            },
            "device": str(input_device),
        }


def build_detection_prompt(
    record: dict,
    retrieved_examples: list[dict],
    calibrated_probability: float,
    risk_band: str,
    graph_features: dict | None = None,
    suspicious_context: dict | None = None,
    semantic_score: float | None = None,
    graph_score: float | None = None,
    fusion_score: float | None = None,
) -> str:
    graph_features = graph_features or {}
    suspicious_context = suspicious_context or {}
    node_info = graph_features.get("node_info") or build_structure_summary(record["code"])
    edge_info = graph_features.get("edge_info") or "Node1\tNode2\tEdgeType\nn/a\tn/a\tSTRUCTURE_SUMMARY_ONLY"
    node_info = truncate_text(node_info, DEFAULT_PROMPT_NODE_INFO_CHAR_LIMIT)
    edge_info = truncate_text(edge_info, DEFAULT_PROMPT_EDGE_INFO_CHAR_LIMIT)
    graph_backend = graph_features.get("backend", "summary")
    suspicious_slices_text = suspicious_context.get("slices_text") or "No suspicious slices."
    suspicious_slices_text = truncate_text(suspicious_slices_text, DEFAULT_PROMPT_SLICES_CHAR_LIMIT)
    top_lines = suspicious_context.get("top_lines") or []
    top_lines_text = "\n".join(
        [
            f"line {row['line_number']}: score={row['score']:.4f} | reasons={','.join(row.get('reasons', [])) or 'n/a'} | {truncate_text(row['code'], DEFAULT_PROMPT_TOP_LINE_CHAR_LIMIT)}"
            for row in top_lines[:DEFAULT_PROMPT_TOP_LINES_LIMIT]
        ]
    )
    token_highlights = ", ".join(suspicious_context.get("token_highlights") or []) or "none"
    example_blocks = []
    for index, example in enumerate(retrieved_examples, start=1):
        label_text = "Vulnerable" if int(example["label"]) == 1 else "Non-vulnerable"
        example_blocks.append(
            "\n".join(
                [
                    f"Reference Example {index}",
                    f"Label: {label_text}",
                    f"Project: {example.get('project', '') or 'unknown'}",
                    "```c",
                    example["code"],
                    "```",
                ]
            )
        )
    examples_text = "\n\n".join(example_blocks) if example_blocks else "No demonstrations."
    return "\n".join(
        [
            "You are auditing one C/C++ function for security vulnerabilities.",
            "Use concrete code evidence only.",
            "Be recall-oriented on real bug patterns, but do not invent vulnerabilities without supporting code evidence.",
            "Focus on memory safety, bounds checks, pointer misuse, lifetime bugs, unsafe APIs, integer overflow, race-prone state changes, and auth or validation flaws.",
            "The demonstrations are similar functions for in-context learning only. Do not copy their labels blindly.",
            f"Prefilter risk band: {risk_band}",
            f"Prefilter calibrated vulnerability probability: {calibrated_probability:.4f}",
            f"Fusion prefilter score: {float(fusion_score or calibrated_probability):.4f}",
            f"Semantic branch score: {float(semantic_score or 0.0):.4f}",
            f"Graph branch score: {float(graph_score or 0.0):.4f}",
            f"Graph backend used for structure extraction: {graph_backend}",
            "",
            "Code snippet:",
            "```c",
            truncate_text(record["code"], DEFAULT_PROMPT_CODE_CHAR_LIMIT),
            "```",
            "",
            "Suspicious lines ranked by the localizer:",
            top_lines_text or "No suspicious lines.",
            "",
            "Suspicious slices with short context:",
            suspicious_slices_text,
            "",
            f"Highlighted local tokens: {token_highlights}",
            "",
            "Use the suspicious slices as guidance, but verify against the full function before deciding.",
            "In the above code snippet, check for potential security vulnerabilities and output either 'Vulnerable' or 'Non-vulnerable'.",
            "The node information of the function is as follows:",
            node_info,
            "",
            "The edge information of the function is as follows:",
            edge_info,
            "",
            "The following are demonstrations retrieved from similar functions:",
            examples_text,
            "",
            "Do not output step-by-step reasoning or any extra commentary.",
            *(
                [
                    "Return exactly one line in this format:",
                    'FINAL_JSON: {"label":"Vulnerable"|"Non-vulnerable","confidence":0.0-1.0,"cwe_family":"CWE-xxx or empty","vulnerable_lines":["start-end"],"sink_or_api":"short name","missing_guard":"short phrase","reason":"short justification"}',
                    "If you decide Vulnerable, fill vulnerable_lines and sink_or_api with concrete evidence; otherwise keep them empty.",
                ]
                if EVIDENCE_SCHEMA_ENABLED
                else [
                    "Return exactly one line in this format:",
                    'FINAL_JSON: {"label":"Vulnerable"|"Non-vulnerable","confidence":0.0-1.0,"reason":"short justification"}',
                ]
            ),
            f"Keep `reason` under {DEFAULT_REASON_WORD_LIMIT} words.",
        ]
    )


def parse_detection_response(text: str) -> dict:
    return _parse_detection_payload(text)


def _parse_detection_payload(raw_text: str) -> dict:
    payload = _extract_json_payload(raw_text)
    if payload is not None:
        return _normalize_response_payload(payload)

    label = _extract_label_fallback(raw_text)
    if label is None:
        raise ValueError("Could not parse a vulnerability label from the local LLM output.")
    confidence = _extract_confidence_fallback(raw_text)
    reason = _extract_reason_fallback(raw_text)
    return _normalize_response_payload(
        {
            "label": label,
            "confidence": confidence,
            "reason": reason,
        }
    )


def _extract_json_payload(raw_text: str) -> dict | None:
    stripped = (raw_text or "").strip()
    if not stripped:
        return None

    candidates = []
    final_json_match = re.search(r"FINAL_JSON:\s*(\{.*\})", stripped, flags=re.IGNORECASE | re.DOTALL)
    if final_json_match:
        candidates.append(final_json_match.group(1).strip())
    candidates.append(stripped)

    for candidate in candidates:
        for value in _iter_json_candidates(candidate):
            if isinstance(value, dict) and EXPECTED_RESPONSE_KEYS.issubset(value.keys()):
                return value
    return None


def _iter_json_candidates(text: str):
    decoder = json.JSONDecoder()
    try:
        yield json.loads(text)
    except Exception:
        pass

    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        yield value


def _normalize_line_spans(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parts = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
        return parts or [text]
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _normalize_response_payload(payload: dict) -> dict:
    label_text = str(payload.get("label", "")).strip()
    normalized = label_text.lower().replace("_", "-")
    if normalized in {"vulnerable", "1"}:
        label_int = 1
        canonical_label = "Vulnerable"
    elif normalized in {"non-vulnerable", "not vulnerable", "non vulnerable", "safe", "benign", "0"}:
        label_int = 0
        canonical_label = "Non-vulnerable"
    else:
        raise ValueError(f"Invalid label in local LLM response: {label_text!r}")

    try:
        confidence = float(payload.get("confidence"))
    except Exception as exc:
        raise ValueError(f"Invalid confidence in local LLM response: {payload.get('confidence')!r}") from exc
    confidence = max(0.0, min(1.0, confidence))

    reason = str(payload.get("reason") or payload.get("brief_reason") or "").strip()
    if not reason:
        raise ValueError("Local LLM response is missing a non-empty reason.")

    return {
        "label": canonical_label,
        "label_int": label_int,
        "confidence": confidence,
        "reason": reason,
        "cwe_family": str(payload.get("cwe_family", "")).strip(),
        "vulnerable_lines": _normalize_line_spans(payload.get("vulnerable_lines")),
        "sink_or_api": str(payload.get("sink_or_api", "")).strip(),
        "missing_guard": str(payload.get("missing_guard", "")).strip(),
    }


def _extract_label_fallback(raw_text: str) -> str | None:
    stripped = (raw_text or "").strip()
    if not stripped:
        return None
    patterns = [
        r"FINAL_LABEL\s*[:=]\s*(VULNERABLE|NON-VULNERABLE|NON VULNERABLE|SAFE|BENIGN)",
        r"final answer\s*[:=]\s*(vulnerable|non-vulnerable|non vulnerable|safe|benign)",
        r"verdict\s*[:=]\s*(vulnerable|non-vulnerable|non vulnerable|safe|benign)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    tail = "\n".join(stripped.splitlines()[-8:])
    if re.search(r"\bnon[- ]vulnerable\b|\bsafe\b|\bbenign\b", tail, flags=re.IGNORECASE):
        return "Non-vulnerable"
    if re.search(r"\bvulnerable\b", tail, flags=re.IGNORECASE):
        return "Vulnerable"
    return None


def _extract_confidence_fallback(raw_text: str) -> float:
    match = re.search(r"confidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", raw_text or "", flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.5


def _extract_reason_fallback(raw_text: str) -> str:
    stripped = (raw_text or "").strip()
    if not stripped:
        return "fallback_parse_empty_output"
    reason_match = re.search(r"reason\s*[:=]\s*(.+)", stripped, flags=re.IGNORECASE)
    if reason_match:
        return truncate_text(reason_match.group(1).strip(), 280) or "fallback_parse_reason_line"
    final_json_prefix = re.sub(r"FINAL_JSON:.*", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    reason_source = final_json_prefix or stripped
    reason_source = re.sub(r"\s+", " ", reason_source).strip()
    return truncate_text(reason_source, 280) or "fallback_parse_short_output"


def _normalize_cache_entry(entry: Any) -> dict:
    if isinstance(entry, str):
        parsed = parse_detection_response(entry)
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_name": None,
            "raw_text": entry,
            "parsed": parsed,
            "finish_reason": None,
            "usage_metadata": {},
            "device": None,
            "cached_at_unix": None,
        }
    if not isinstance(entry, dict):
        raise ValueError(f"Unsupported cache entry type: {type(entry).__name__}")
    parsed = _parse_detection_payload(entry.get("raw_text", ""))
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": entry.get("model_name"),
        "raw_text": entry.get("raw_text", ""),
        "parsed": parsed,
        "finish_reason": entry.get("finish_reason"),
        "usage_metadata": entry.get("usage_metadata", {}),
        "device": entry.get("device"),
        "cached_at_unix": entry.get("cached_at_unix"),
    }
