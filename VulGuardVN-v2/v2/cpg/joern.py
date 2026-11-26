"""Batch Joern runner (plan P2 step 1): ~1,000 functions per Joern project, several JVMs in parallel.

Output: `artifacts/{dataset}/cpg_cache/facts.jsonl` (one JSON line per function id, resumable).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..config import CPG_SCRIPT_DIR, dataset_dir, get_logger, load_config, resolve_project_path

log = get_logger("p2.joern")
_LOCK = threading.Lock()

# Kernel / compiler annotations that the CDT parser in c2cpg cannot resolve without headers. They are blanked
# with spaces of the same length (newlines kept) so Joern line numbers stay identical to the LLM view.
_ANNOTATIONS = re.compile(
    r"\b(?:__user|__kernel|__iomem|__force|__rcu|__percpu|__init|__exit|__initdata|__devinit|__devexit|__cold|"
    r"__hot|__must_check|__maybe_unused|__always_inline|noinline|asmlinkage|__visible|__weak|__packed|"
    r"__read_mostly|__latent_entropy|__noreturn|__bitwise|__deprecated|__sched|__ref|__used|"
    r"__aligned\s*\([^)]*\)|__printf\s*\([^)]*\)|__acquires\s*\([^)]*\)|__releases\s*\([^)]*\)|"
    r"__must_hold\s*\([^)]*\)|__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\))")
_CPP_HINT = re.compile(r"::|\bstd::|\bnullptr\b|\btemplate\s*<|\bnamespace\b|\bclass\s+\w+|&\s*\w+\s*[,)]")
_NO_RETURN_TYPE = re.compile(r"^[A-Za-z_]\w*\s*\(")


def detect_lang(code: str, declared: str) -> str:
    if declared == "cpp":
        return "cpp"
    return "cpp" if _CPP_HINT.search(code[:800]) else "c"


def joern_source(code: str) -> str:
    """Copy of the function given to Joern. Same number of lines as `code` (line numbers must match)."""
    text = _ANNOTATIONS.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), code)
    first, _, rest = text.partition("\n")
    if _NO_RETURN_TYPE.match(first):          # return type was on a line that the dataset cut off
        first = "int " + first
    return first + ("\n" + rest if rest else "")


def joern_env(cfg: dict) -> tuple[Path, dict]:
    joern_home = resolve_project_path(cfg["joern_home"])
    java_home = resolve_project_path(cfg["java_home"])
    executable = joern_home / "joern"
    if not executable.exists():
        raise FileNotFoundError(f"Joern not found at {executable} - see README (P0 step 3)")
    env = dict(os.environ)
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = f"{java_home / 'bin'}:{env.get('PATH', '')}"
    return executable, env


def joern_version(cfg: dict) -> str:
    joern_home = resolve_project_path(cfg["joern_home"])
    jars = sorted((joern_home / "lib").glob("io.joern.semanticcpg-*.jar"))
    return jars[0].stem.split("-")[-1] if jars else "unknown"


def facts_path(dataset: str) -> Path:
    return dataset_dir(dataset, "cpg_cache") / "facts.jsonl"


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


def _run_batch(batch: pd.DataFrame, cfg: dict, work_root: Path, tag: str, flip_lang: bool = False) -> list[dict]:
    executable, env = joern_env(cfg)
    work = work_root / tag
    shutil.rmtree(work, ignore_errors=True)
    in_dir = work / "in"
    in_dir.mkdir(parents=True)
    stem_to_id: dict[str, str] = {}
    lang_of: dict[str, str] = {}
    for index, row in enumerate(batch.itertuples(index=False)):
        stem = f"{index:05d}"
        lang = detect_lang(row.code, row.lang)
        if flip_lang:
            lang = "c" if lang == "cpp" else "cpp"
        (in_dir / f"{stem}{'.cpp' if lang == 'cpp' else '.c'}").write_text(joern_source(row.code) + "\n", encoding="utf-8")
        stem_to_id[stem] = row.id
        lang_of[row.id] = lang
    out_file = work / "out.jsonl"
    command = [str(executable), f"-J-Xmx{cfg.get('jvm_heap', '8g')}", "--script", str(CPG_SCRIPT_DIR / "extract.sc"),
               "--param", f"inputDir={in_dir}", "--param", f"outFile={out_file}",
               "--param", "sinkApis=" + ",".join(cfg["sink_apis"]),
               "--param", "allocApis=" + ",".join(cfg["alloc_apis"]),
               "--param", "sourceApis=" + ",".join(cfg["source_apis"]),
               "--param", f"maxSinks={int(cfg.get('max_candidate_sinks', 30))}",
               "--param", f"maxSlice={int(cfg.get('max_slice_nodes', 400))}"]
    error = None
    try:
        proc = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True,
                              timeout=int(cfg.get("batch_timeout_s", 3600)))
        if proc.returncode != 0:
            error = f"exit_{proc.returncode}: {proc.stderr[-400:]}"
    except subprocess.TimeoutExpired:
        error = "timeout"
    results: dict[str, dict] = {}
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            stem = payload.get("id")
            if stem in stem_to_id:
                payload["id"] = stem_to_id[stem]
                payload["slices"] = [sorted(set(s)) for s in payload.get("slices", [])]
                payload["lang"] = lang_of[payload["id"]]
                payload["attempt"] = 2 if flip_lang else 1
                results[payload["id"]] = payload
    missing = [stem_to_id[s] for s in stem_to_id if stem_to_id[s] not in results]
    shutil.rmtree(work, ignore_errors=True)
    if missing and error and len(batch) > 25:
        # crash/timeout: retry the missing part in two halves
        rest = batch[batch["id"].isin(missing)]
        half = len(rest) // 2
        log.info("batch %s failed (%s) - retrying %d functions in halves", tag, error[:80], len(rest))
        return list(results.values()) + _run_batch(rest.iloc[:half], cfg, work_root, tag + "a", flip_lang) + \
            _run_batch(rest.iloc[half:], cfg, work_root, tag + "b", flip_lang)
    for record_id in missing:
        results[record_id] = {"id": record_id, "ok": False, "error": error or "missing_output",
                              "attempt": 2 if flip_lang else 1}
    return list(results.values())


def run_cpg(dataset: str, targets: pd.DataFrame, cfg: dict | None = None, limit: int | None = None) -> dict:
    cfg = cfg or load_config("cpg")
    out_path = facts_path(dataset)
    done = load_done_ids(out_path)
    todo = targets[~targets["id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    size = int(cfg.get("batch_size", 1000))
    batches = [todo.iloc[i: i + size] for i in range(0, len(todo), size)]
    log.info("[%s] CPG: %d targets, %d done, %d to run in %d batches (workers=%d)", dataset, len(targets), len(done),
             len(todo), len(batches), int(cfg.get("workers", 2)))
    work_root = dataset_dir(dataset, "cpg_cache", "work")
    started = time.perf_counter()
    _run_all(dataset, batches, cfg, work_root, out_path, "b", flip_lang=False, total=len(todo))
    # second attempt for parse failures: the other C/C++ frontend mode
    latest = load_facts(dataset, set(targets["id"]))
    retry_ids = [i for i, f in latest.items() if not f.get("ok") and f.get("attempt", 1) == 1
                 and f.get("error") in {"no_method", "missing_output"}]
    if retry_ids and not limit:
        retry = targets[targets["id"].isin(retry_ids)]
        log.info("[%s] retrying %d parse failures with the other language mode", dataset, len(retry))
        retry_batches = [retry.iloc[i: i + size] for i in range(0, len(retry), size)]
        _run_all(dataset, retry_batches, cfg, work_root, out_path, "r", flip_lang=True, total=len(retry), keep_failures=False)
    return {"ran": int(len(todo)), "seconds": time.perf_counter() - started}


def _run_all(dataset, batches, cfg, work_root, out_path, prefix, *, flip_lang, total, keep_failures=True) -> None:
    started = time.perf_counter()
    processed = 0
    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 2))) as pool:
        futures = {pool.submit(_run_batch, batch, cfg, work_root, f"{prefix}{index:04d}", flip_lang): len(batch)
                   for index, batch in enumerate(batches)}
        for future in as_completed(futures):
            rows = future.result()
            with _LOCK, out_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    if row.get("ok") or keep_failures:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:   # keep the first failure but mark that the retry also failed
                        handle.write(json.dumps({**row, "attempt": 2}, ensure_ascii=False) + "\n")
            processed += futures[future]
            ok = sum(1 for row in rows if row.get("ok"))
            log.info("[%s] batch done: %d/%d ok | total %d/%d | %.1f fn/min", dataset, ok, len(rows), processed,
                     total, 60 * processed / max(time.perf_counter() - started, 1e-6))


def load_facts(dataset: str, ids: set[str] | None = None) -> dict[str, dict]:
    path = facts_path(dataset)
    facts: dict[str, dict] = {}
    if not path.exists():
        return facts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:          # interrupted append - that function is simply re-run
                continue
            if ids is None or payload["id"] in ids:
                facts[payload["id"]] = payload
    return facts
