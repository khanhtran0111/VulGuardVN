import json
import importlib
import re
import time
from pathlib import Path
from typing import Any

from common import CACHE_DIR, build_structure_summary, ensure_dir, load_json, resolve_gemini_api_key, stable_hash, truncate_text


CACHE_SCHEMA_VERSION = 2
EXPECTED_RESPONSE_KEYS = {"label", "confidence", "reason"}
DETECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string", "enum": ["Vulnerable", "Non-vulnerable"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "required": ["label", "confidence", "reason"],
}

_SDK_KIND = None
_genai_module = None
_genai_types = None
_legacy_genai = None
_sdk_import_error = None

try:
    _genai_module = importlib.import_module("google.genai")
    _genai_types = importlib.import_module("google.genai.types")
    _SDK_KIND = "google-genai"
except Exception as exc:
    _sdk_import_error = exc
    try:
        _legacy_genai = importlib.import_module("google.generativeai")
        _SDK_KIND = "google-generativeai"
    except Exception as legacy_exc:
        _sdk_import_error = legacy_exc


class GeminiClassifier:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 512,
        cache_path: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.client = _build_gemini_client(resolve_gemini_api_key())
        self.cache_path = cache_path or (CACHE_DIR / "gemini_cache.json")
        ensure_dir(self.cache_path.parent)
        self.cache = load_json(self.cache_path, default={})

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cache_key(self, prompt: str) -> str:
        return stable_hash(f"{self.model_name}\n{prompt}")

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

    def classify(self, prompt: str) -> dict:
        cached_entry = self._get_cached_entry(prompt)
        if cached_entry is not None:
            result = dict(cached_entry["parsed"])
            result["raw_text"] = cached_entry["raw_text"]
            result["cached"] = True
            result["finish_reason"] = cached_entry.get("finish_reason")
            result["usage_metadata"] = cached_entry.get("usage_metadata", {})
            return result

        cache_key = self._cache_key(prompt)
        last_error = None
        for attempt in range(self.max_retries):
            response_payload = None
            try:
                response_payload = _generate_response_payload(
                    client=self.client,
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                parsed = _parse_detection_payload(response_payload.get("parsed"), response_payload.get("raw_text", ""))
                cache_entry = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "model_name": self.model_name,
                    "raw_text": response_payload.get("raw_text", ""),
                    "parsed": parsed,
                    "finish_reason": response_payload.get("finish_reason"),
                    "usage_metadata": response_payload.get("usage_metadata", {}),
                    "cached_at_unix": time.time(),
                }
                self.cache[cache_key] = cache_entry
                self._save_cache()
                result = dict(parsed)
                result["raw_text"] = cache_entry["raw_text"]
                result["cached"] = False
                result["finish_reason"] = cache_entry["finish_reason"]
                result["usage_metadata"] = cache_entry["usage_metadata"]
                return result
            except Exception as exc:
                if response_payload is not None:
                    finish_reason = response_payload.get("finish_reason")
                    raw_preview = (response_payload.get("raw_text", "") or "")[:160]
                    last_error = RuntimeError(
                        f"{exc} | finish_reason={finish_reason} | raw_text_preview={raw_preview!r}"
                    )
                else:
                    last_error = exc
                self._invalidate_cache_key(cache_key)
                if attempt >= self.max_retries - 1 or not _should_retry_exception(last_error):
                    break
                time.sleep(_retry_sleep_seconds(last_error, attempt))
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_error}")


def build_detection_prompt(record: dict, retrieved_examples: list[dict], calibrated_probability: float, risk_band: str) -> str:
    example_blocks = []
    for index, example in enumerate(retrieved_examples, start=1):
        label_text = "Vulnerable" if int(example["label"]) == 1 else "Non-vulnerable"
        example_blocks.append(
            "\n".join(
                [
                    f"Example {index}",
                    f"Label: {label_text}",
                    f"Project: {example.get('project', '') or 'unknown'}",
                    "Structure cues:",
                    example["structure_summary"],
                    "Code:",
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
            'Return JSON only with keys label, confidence, reason. label must be exactly "Vulnerable" or "Non-vulnerable".',
            'Do not add markdown fences, backticks, or any prose before/after the JSON object.',
            "Use concrete code evidence only. Focus on memory safety, bounds checks, pointer misuse, lifetime bugs, unsafe API usage, integer overflow, race-prone state changes, auth or validation flaws.",
            f"Prefilter risk band: {risk_band}",
            f"Prefilter calibrated vulnerability probability: {calibrated_probability:.4f}",
            "The demonstrations are similar reference cases, not ground truth for the target.",
            "",
            "Reference demonstrations:",
            examples_text,
            "",
            "Target structure cues:",
            build_structure_summary(record["code"]),
            "",
            "Target function:",
            "```c",
            truncate_text(record["code"], 6000),
            "```",
        ]
    )


def parse_detection_response(text: str) -> dict:
    return _parse_detection_payload(None, text)


def _parse_detection_payload(parsed_payload: Any, raw_text: str) -> dict:
    payload = parsed_payload if isinstance(parsed_payload, dict) else None
    if payload is None:
        payload = _parse_json_text(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Gemini returned non-object payload: {type(payload).__name__}")

    label_text = str(payload.get("label", "")).strip()
    normalized = label_text.lower().replace("_", "-")
    if normalized in {"vulnerable", "1"}:
        label_int = 1
        canonical_label = "Vulnerable"
    elif normalized in {"non-vulnerable", "not vulnerable", "non vulnerable", "0"}:
        label_int = 0
        canonical_label = "Non-vulnerable"
    else:
        raise ValueError(f"Invalid label in Gemini response: {label_text!r}")

    try:
        confidence = float(payload.get("confidence"))
    except Exception as exc:
        raise ValueError(f"Invalid confidence in Gemini response: {payload.get('confidence')!r}") from exc
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Confidence out of range in Gemini response: {confidence}")

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("Gemini response is missing a non-empty reason.")

    return {
        "label": canonical_label,
        "label_int": label_int,
        "confidence": confidence,
        "reason": reason,
    }


def _parse_json_text(raw_text: str) -> dict:
    stripped = (raw_text or "").strip()
    if not stripped:
        raise ValueError("Gemini response text is empty.")

    first_dict = None
    for candidate in _iter_json_payload_candidates(stripped):
        for value in _iter_decoded_json_values(candidate):
            if not isinstance(value, dict):
                continue
            if EXPECTED_RESPONSE_KEYS.issubset(value.keys()):
                return value
            if first_dict is None:
                first_dict = value
    if first_dict is not None:
        return first_dict
    raise ValueError(f"Gemini response is not valid JSON: {stripped[:160]!r}")


def _iter_json_payload_candidates(text: str):
    seen = set()

    def _push(value: str):
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        return normalized

    direct = _push(text)
    if direct is not None:
        yield direct

    stripped_fence = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = _push(stripped_fence)
    if cleaned is not None:
        yield cleaned

    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL):
        block = _push(match.group(1))
        if block is not None:
            yield block


def _iter_decoded_json_values(text: str):
    decoder = json.JSONDecoder()
    try:
        yield json.loads(text)
    except Exception:
        pass

    for match in re.finditer(r"[\{\[]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        yield value


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
            "cached_at_unix": None,
        }
    if not isinstance(entry, dict):
        raise ValueError(f"Unsupported cache entry type: {type(entry).__name__}")
    parsed = _parse_detection_payload(entry.get("parsed"), entry.get("raw_text", ""))
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": entry.get("model_name"),
        "raw_text": entry.get("raw_text", ""),
        "parsed": parsed,
        "finish_reason": entry.get("finish_reason"),
        "usage_metadata": _to_jsonable(entry.get("usage_metadata", {})),
        "cached_at_unix": entry.get("cached_at_unix"),
    }


def _retry_sleep_seconds(exc: Exception, attempt: int) -> float:
    message = str(exc)
    retry_delay_match = re.search(r"retryDelay[\"'=:\s]+(?:(\d+)(?:\.\d+)?)s", message, re.IGNORECASE)
    if retry_delay_match:
        return max(5.0, float(retry_delay_match.group(1)) + 1.0)
    lowered = message.lower()
    if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered or "rate limit" in lowered:
        return min(90.0, 15.0 * (attempt + 1))
    if "503" in lowered or "unavailable" in lowered:
        return min(30.0, 5.0 * (attempt + 1))
    return 1.5 * (attempt + 1)


def _should_retry_exception(exc: Exception) -> bool:
    lowered = str(exc).lower()
    if "generate_content_free_tier_requests" in lowered:
        return False
    if "perdayperprojectpermodel" in lowered:
        return False
    if "check your plan and billing details" in lowered:
        return False
    if "api key not valid" in lowered or "permission_denied" in lowered or "unauthenticated" in lowered:
        return False
    return True


def _build_gemini_client(api_key: str):
    if _SDK_KIND == "google-genai":
        return _genai_module.Client(api_key=api_key)
    if _SDK_KIND == "google-generativeai":
        _legacy_genai.configure(api_key=api_key)
        return _legacy_genai.GenerativeModel
    raise RuntimeError(
        "Could not import a Gemini Python SDK. "
        "Install `google-genai` (preferred) or `google-generativeai`. "
        f"Last import error: {_sdk_import_error}"
    )


def _generate_response_payload(client, model_name: str, prompt: str, temperature: float, max_output_tokens: int) -> dict:
    if _SDK_KIND == "google-genai":
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=DETECTION_SCHEMA,
                thinking_config=_genai_types.ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=0,
                ),
            ),
        )
        return {
            "raw_text": response.text or "",
            "parsed": _to_jsonable(getattr(response, "parsed", None)),
            "finish_reason": _extract_finish_reason(response),
            "usage_metadata": _to_jsonable(getattr(response, "usage_metadata", None)),
        }
    if _SDK_KIND == "google-generativeai":
        model = client(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        return {
            "raw_text": getattr(response, "text", "") or "",
            "parsed": None,
            "finish_reason": None,
            "usage_metadata": {},
        }
    raise RuntimeError("Gemini SDK is not available.")


def _extract_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return None
    return str(finish_reason)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(exclude_none=True))
    if hasattr(value, "to_json_dict"):
        return _to_jsonable(value.to_json_dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)
