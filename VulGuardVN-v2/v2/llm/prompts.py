"""Prompt rendering and prompt hashing (D6: prompts are frozen by hash before A-pool / test)."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import PROMPT_DIR, SCHEMA_DIR

# stage -> (template, schema). Ablation / RQ5 stages reuse the frozen templates of the main stages.
STAGES = {
    "pass1": ("pass1.j2", "pass1.json"),
    "pass2_neutral": ("pass2_neutral.j2", "pass2.json"),
    "pass2_refute": ("pass2_refute.j2", "pass2.json"),
    "zeroshot": ("zeroshot.j2", "zeroshot.json"),
}
FROZEN_STAGES = tuple(STAGES)


def base_stage(stage: str) -> str:
    """a7_neutral -> pass2_neutral, rq5_refute -> pass2_refute, repeat_pass1 -> pass1 ..."""
    for name in STAGES:
        if stage == name:
            return name
    if stage.endswith("neutral"):
        return "pass2_neutral"
    if stage.endswith("refute"):
        return "pass2_refute"
    if stage.endswith("pass1"):
        return "pass1"
    if stage.endswith("zeroshot"):
        return "zeroshot"
    raise KeyError(stage)


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(PROMPT_DIR)), undefined=StrictUndefined, trim_blocks=False,
                       keep_trailing_newline=False, autoescape=False)


@lru_cache(maxsize=None)
def schema(stage: str) -> dict:
    return json.loads((SCHEMA_DIR / STAGES[base_stage(stage)][1]).read_text(encoding="utf-8"))


def prompt_hash(stage: str) -> str:
    template, schema_file = STAGES[base_stage(stage)]
    blob = (PROMPT_DIR / "system.j2").read_text(encoding="utf-8") + "\n--\n" + \
        (PROMPT_DIR / template).read_text(encoding="utf-8") + "\n--\n" + (SCHEMA_DIR / schema_file).read_text(encoding="utf-8")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def current_hashes() -> dict[str, str]:
    return {stage: prompt_hash(stage) for stage in FROZEN_STAGES}


def render(stage: str, **variables) -> list[dict]:
    env = _env()
    system = env.get_template("system.j2").render().strip()
    user = env.get_template(STAGES[base_stage(stage)][0]).render(**variables).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
