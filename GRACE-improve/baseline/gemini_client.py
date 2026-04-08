import json
import importlib
import re
import time
from pathlib import Path

from common import CACHE_DIR, build_structure_summary, ensure_dir, load_json, resolve_gemini_api_key, stable_hash, truncate_text


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
        max_output_tokens: int = 256,
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

    def is_cached(self, prompt: str) -> bool:
        return self._cache_key(prompt) in self.cache

    def generate(self, prompt: str) -> str:
        cache_key = self._cache_key(prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        last_error = None
        for attempt in range(self.max_retries):
            try:
                text = _generate_text(
                    client=self.client,
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                self.cache[cache_key] = text
                self._save_cache()
                return text
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries - 1:
                    break
                time.sleep(_retry_sleep_seconds(exc, attempt))
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_error}")

    def classify(self, prompt: str) -> dict:
        raw_text = self.generate(prompt)
        parsed = parse_detection_response(raw_text)
        parsed["raw_text"] = raw_text
        return parsed


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
            'Return JSON only: {"label":"Vulnerable"|"Non-vulnerable","confidence":0.0-1.0,"reason":"short justification"}.',
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
    stripped = (text or "").strip()
    if not stripped:
        return {"label": "Non-vulnerable", "label_int": 0, "confidence": 0.0, "reason": "empty response"}
    payload = None
    try:
        payload = json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    if isinstance(payload, dict):
        label_text = str(payload.get("label", "")).strip()
        confidence = payload.get("confidence", 0.0)
        reason = str(payload.get("reason", "")).strip()
    else:
        label_text = stripped
        confidence = 0.0
        reason = stripped[:200]
    normalized = label_text.lower()
    if "non-vulnerable" in normalized or "not vulnerable" in normalized or normalized == "0":
        label = 0
        canonical = "Non-vulnerable"
    elif "vulnerable" in normalized or normalized == "1":
        label = 1
        canonical = "Vulnerable"
    else:
        label = 0
        canonical = "Non-vulnerable"
    try:
        confidence_value = float(confidence)
    except Exception:
        confidence_value = 0.0
    confidence_value = max(0.0, min(1.0, confidence_value))
    return {
        "label": canonical,
        "label_int": label,
        "confidence": confidence_value,
        "reason": reason,
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


def _generate_text(client, model_name: str, prompt: str, temperature: float, max_output_tokens: int) -> str:
    if _SDK_KIND == "google-genai":
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return response.text or ""
    if _SDK_KIND == "google-generativeai":
        model = client(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        return getattr(response, "text", "") or ""
    raise RuntimeError("Gemini SDK is not available.")
