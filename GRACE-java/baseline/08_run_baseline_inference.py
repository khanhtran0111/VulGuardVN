from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from common import WORK_DIR, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline LLM inference over assembled prompts.")
    p.add_argument("--input", default=str(WORK_DIR / "test_prompts.jsonl"))
    p.add_argument("--output", default=str(WORK_DIR / "test_predictions.jsonl"))
    p.add_argument("--provider", choices=["dry_run", "openai", "gemini"], default="dry_run")
    p.add_argument("--model", default=None, help="Model name. If omitted, provider-specific defaults are used.")
    return p.parse_args()


def load_simple_dotenv(dotenv_path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    if not dotenv_path.exists():
        return env_map

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        env_map[key] = value
    return env_map


def parse_binary_label(text: str) -> int:
    lowered = text.lower()
    if "non-vulnerable" in lowered or "not vulnerable" in lowered:
        return 0
    if "vulnerable" in lowered:
        return 1
    return -1


def call_openai(prompt: str, model: str) -> str:
    # Lazy import so the script still works in dry-run mode without the package.
    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        temperature=0,
    )
    return resp.output_text


def call_gemini(prompt: str, model: str, api_key: str) -> str:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urlerror.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini HTTPError {e.code}: {detail}") from e

    candidates = body.get("candidates") or []
    if not candidates:
        return ""

    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
    return "\n".join(t for t in texts if t).strip()


def resolve_model(provider: str, cli_model: str | None, dotenv_vars: dict[str, str]) -> str:
    if cli_model:
        return cli_model
    if provider == "openai":
        return "gpt-4.1"
    if provider == "gemini":
        return dotenv_vars.get("GEMINI_MODEL") or "gemini-2.5-flash"
    return "dry_run"


def main() -> None:
    args = parse_args()
    repo_root = WORK_DIR.parent
    dotenv_vars = load_simple_dotenv(repo_root / ".env")

    rows = load_jsonl(args.input)
    outputs = []
    model_name = resolve_model(args.provider, args.model, dotenv_vars)

    if args.provider == "gemini":
        gemini_api_key = os.environ.get("GEMINI_API") or dotenv_vars.get("GEMINI_API")
        if not gemini_api_key:
            raise ValueError("Missing GEMINI_API. Set env var or add GEMINI_API in GRACE-java/.env")
    else:
        gemini_api_key = None

    for row in rows:
        prompt = row["prompt"]
        if args.provider == "dry_run":
            raw = "DRY_RUN"
            pred = -1
        elif args.provider == "openai":
            raw = call_openai(prompt, model_name)
            pred = parse_binary_label(raw)
        else:
            raw = call_gemini(prompt, model_name, gemini_api_key)
            pred = parse_binary_label(raw)

        out = {
            "raw_response": raw,
            "pred_label": pred,
        }
        outputs.append(out)

    write_jsonl(args.output, outputs)
    print(json.dumps({"num_rows": len(outputs), "output": str(Path(args.output))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
