from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import WORK_DIR, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline LLM inference over assembled prompts.")
    p.add_argument("--input", default=str(WORK_DIR / "test_prompts.jsonl"))
    p.add_argument("--output", default=str(WORK_DIR / "test_predictions.jsonl"))
    p.add_argument("--provider", choices=["dry_run", "openai"], default="dry_run")
    p.add_argument("--model", default="gpt-4.1")
    return p.parse_args()


def parse_binary_label(text: str) -> int:
    lowered = text.lower()
    if "non-vulnerable" in lowered or "not vulnerable" in lowered:
        return 0
    if "vulnerable" in lowered:
        return 1
    return -1


def call_openai(prompt: str, model: str) -> str:
    # Lazy import so the script still works in dry-run mode without the package.
    from openai import OpenAI

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


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    outputs = []

    for row in rows:
        prompt = row["prompt"]
        if args.provider == "dry_run":
            raw = "DRY_RUN"
            pred = -1
        else:
            raw = call_openai(prompt, args.model)
            pred = parse_binary_label(raw)

        out = dict(row)
        out["model_name"] = args.model
        out["raw_response"] = raw
        out["pred_label"] = pred
        outputs.append(out)

    write_jsonl(args.output, outputs)
    print(json.dumps({"num_rows": len(outputs), "output": str(Path(args.output))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
