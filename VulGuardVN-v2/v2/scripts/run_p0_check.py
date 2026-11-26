"""P0 - environment check: Python/torch/GPU, Joern + Java, UniXcoder, datasets, API key (never printed).

    python -m v2.scripts.run_p0_check
Writes artifacts/environment.json (versions for the reproducibility checklist, plan §11).
"""
from __future__ import annotations

import os
import platform
import subprocess

from ..common import dump_json
from ..config import PROJECT_DIR, artifacts_dir, get_logger, load_config, resolve_project_path
from ..cpg.joern import joern_env, joern_version

log = get_logger("p0")


def main() -> None:
    import torch
    import transformers
    import lightgbm
    import sklearn

    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "transformers": transformers.__version__, "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__}
    cpg = load_config("cpg")
    try:
        executable, env = joern_env(cpg)
        java = subprocess.run([str(resolve_project_path(cpg["java_home"]) / "bin" / "java"), "-version"], capture_output=True, text=True)
        info["java"] = (java.stderr or java.stdout).splitlines()[0]
        info["joern"] = joern_version(cpg)
        info["joern_ok"] = executable.exists()
    except Exception as exc:  # noqa: BLE001
        info["joern_ok"] = False
        info["joern_error"] = str(exc)
    ucfg = load_config("prefilter")["features"]["unixcoder"]
    info["unixcoder_model"] = str(resolve_project_path(ucfg["model_dir"]))
    info["unixcoder_ok"] = (resolve_project_path(ucfg["model_dir"]) / "config.json").exists()
    dcfg = load_config("datasets")
    root = resolve_project_path(dcfg["data_root"])
    files = {}
    for name, cfg in dcfg["datasets"].items():
        paths = list((cfg.get("files") or {}).values()) + ([cfg["file"]] if "file" in cfg else []) + list((cfg.get("paired_files") or {}).values())
        files[name] = all((root / p).exists() for p in paths)
    info["datasets_present"] = files
    llm = load_config("llm")
    key = os.getenv(llm.get("api_key_env", "OPENROUTER_API_KEY"), "")
    info["openrouter_key_set"] = bool(key.strip())
    info["llm_provider"] = llm["provider"]
    info["prompt_hashes_frozen"] = all(llm.get("prompt_hashes", {}).values())
    dump_json(artifacts_dir() / "environment.json", info)
    for key_name, value in info.items():
        log.info("%-22s %s", key_name, value)
    if not info["openrouter_key_set"]:
        log.info("-> fill OPENROUTER_API_KEY in %s before P4 (P0 step 2)", PROJECT_DIR / ".env")


if __name__ == "__main__":
    main()
