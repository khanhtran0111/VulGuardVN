from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import WORK_DIR, ensure_dir, load_jsonl, slugify, wrap_java_method, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Joern for each Java sample and export AST/CFG/PDG dot files.")
    p.add_argument("--input", default=str(WORK_DIR / "test_with_demo.jsonl"))
    p.add_argument("--output", default=str(WORK_DIR / "test_with_joern.jsonl"))
    p.add_argument("--joern-parse", default="joern-parse")
    p.add_argument("--joern-export", default="joern-export")
    p.add_argument("--keep-temp", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def first_dot_file(directory: Path) -> str | None:
    dots = sorted(directory.rglob("*.dot"))
    return str(dots[0]) if dots else None


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    out_rows = []
    base_out = ensure_dir(Path(args.output).parent / (Path(args.output).stem + "_joern"))

    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", idx))
        sample_slug = slugify(sample_id)
        workdir = Path(tempfile.mkdtemp(prefix=f"joern_{sample_slug}_"))
        srcdir = ensure_dir(workdir / "src")
        class_name = f"Wrap_{sample_slug}"
        java_path = srcdir / f"{class_name}.java"
        java_path.write_text(wrap_java_method(row["code"], class_name), encoding="utf-8")

        cpg_path = workdir / "cpg.bin.zip"
        ast_out = ensure_dir(workdir / "ast")
        cfg_out = ensure_dir(workdir / "cfg")
        pdg_out = ensure_dir(workdir / "pdg")

        status = "ok"
        error = None
        try:
            run_cmd([args.joern_parse, str(srcdir), "--language", "JAVASRC", "--output", str(cpg_path)])
            run_cmd([args.joern_export, str(cpg_path), "--repr", "ast", "--format", "dot", "--out", str(ast_out)])
            run_cmd([args.joern_export, str(cpg_path), "--repr", "cfg", "--format", "dot", "--out", str(cfg_out)])
            run_cmd([args.joern_export, str(cpg_path), "--repr", "pdg", "--format", "dot", "--out", str(pdg_out)])
        except Exception as e:  # noqa: BLE001
            status = "failed"
            error = str(e)

        sample_out = ensure_dir(base_out / sample_slug)
        if status == "ok":
            ast_file = first_dot_file(ast_out)
            cfg_file = first_dot_file(cfg_out)
            pdg_file = first_dot_file(pdg_out)
            if ast_file:
                shutil.copy2(ast_file, sample_out / "ast.dot")
            if cfg_file:
                shutil.copy2(cfg_file, sample_out / "cfg.dot")
            if pdg_file:
                shutil.copy2(pdg_file, sample_out / "pdg.dot")
        else:
            ast_file = cfg_file = pdg_file = None

        out = dict(row)
        out["joern_status"] = status
        out["joern_error"] = error
        out["joern_ast_dot"] = str(sample_out / "ast.dot") if (sample_out / "ast.dot").exists() else None
        out["joern_cfg_dot"] = str(sample_out / "cfg.dot") if (sample_out / "cfg.dot").exists() else None
        out["joern_pdg_dot"] = str(sample_out / "pdg.dot") if (sample_out / "pdg.dot").exists() else None
        out_rows.append(out)

        if args.keep_temp:
            print(f"[{sample_id}] temp dir kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    write_jsonl(args.output, out_rows)
    ok = sum(1 for r in out_rows if r["joern_status"] == "ok")
    print(f"Finished Joern export. ok={ok}/{len(out_rows)} | output={args.output}")


if __name__ == "__main__":
    main()
