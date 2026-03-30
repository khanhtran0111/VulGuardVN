from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import WORK_DIR, ensure_dir, load_jsonl, slugify, wrap_java_method, write_jsonl


def candidate_suffixes() -> tuple[str, ...]:
    # On Windows, prefer launcher scripts/executables instead of extensionless Unix shell scripts.
    if os.name == "nt":
        return (".bat", ".cmd", ".exe")
    return ("", ".sh", ".bat", ".cmd", ".exe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Joern for each Java sample and export AST/CFG/PDG dot files.")
    p.add_argument("--input", default=str(WORK_DIR / "test_with_demo.jsonl"))
    p.add_argument("--output", default=str(WORK_DIR / "test_with_joern.jsonl"))
    p.add_argument("--joern-parse", default="joern-parse")
    p.add_argument("--joern-export", default="joern-export")
    p.add_argument(
        "--java-home",
        default=None,
        help="Optional JAVA_HOME to run Joern with (must be Java 11+).",
    )
    p.add_argument("--keep-temp", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    run_target = cmd
    if os.name == "nt" and cmd and cmd[0].lower().endswith((".bat", ".cmd")):
        # On Windows, batch scripts must be executed via cmd.exe.
        run_target = ["cmd", "/c", *cmd]

    proc = subprocess.run(run_target, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")


def parse_java_major(version_text: str) -> int | None:
    # Handles both:
    # - java version "1.8.0_..."  -> 8
    # - openjdk version "11.0.22" -> 11
    m = re.search(r'version\s+"([^"]+)"', version_text)
    if not m:
        return None

    raw = m.group(1)
    if raw.startswith("1."):
        parts = raw.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return None

    first = raw.split(".", 1)[0]
    return int(first) if first.isdigit() else None


def java_major_from_cmd(cmd: list[str], env: dict[str, str] | None = None) -> int | None:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    except Exception:  # noqa: BLE001
        return None
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    return parse_java_major(text)


def java_version_line(env: dict[str, str] | None = None) -> str:
    try:
        proc = subprocess.run(["java", "-version"], check=False, capture_output=True, text=True, env=env)
    except Exception:  # noqa: BLE001
        return "unknown"
    text = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip().splitlines()
    return text[0].strip() if text else "unknown"


def java_executable_path(env: dict[str, str] | None = None) -> str:
    if env and env.get("JAVA_HOME"):
        candidate = Path(env["JAVA_HOME"]) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if candidate.exists():
            return str(candidate)

    found = shutil.which("java", path=(env or {}).get("PATH"))
    return found or "java"


def build_env_with_java_home(java_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    java_bin = str(java_home / "bin")
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = java_bin + os.pathsep + env.get("PATH", "")
    return env


def resolve_java_env(cli_java_home: str | None) -> dict[str, str]:
    # 0) Explicit --java-home
    if cli_java_home:
        candidate = Path(cli_java_home).expanduser().resolve()
        env = build_env_with_java_home(candidate)
        major = java_major_from_cmd(["java", "-version"], env=env)
        if major is None or major < 11:
            raise RuntimeError(
                f"Invalid --java-home ({candidate}). Found Java {major}, but Joern requires Java 11+."
            )
        return env

    # 1) Current shell Java
    current_major = java_major_from_cmd(["java", "-version"], env=None)
    if current_major is not None and current_major >= 11:
        return dict(os.environ)

    # 2) Candidate env vars
    env_var_candidates = ["JAVA_HOME", "JDK_HOME", "JAVA11_HOME", "JAVA17_HOME", "JAVA21_HOME"]
    for var in env_var_candidates:
        raw = os.environ.get(var)
        if not raw:
            continue
        candidate = Path(raw)
        if not (candidate / "bin" / ("java.exe" if os.name == "nt" else "java")).exists():
            continue
        env = build_env_with_java_home(candidate)
        major = java_major_from_cmd(["java", "-version"], env=env)
        if major is not None and major >= 11:
            return env

    # 3) Windows default install locations
    if os.name == "nt":
        roots = [Path("C:/Program Files/Java"), Path("C:/Program Files/Eclipse Adoptium")]
        discovered: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir() and (child / "bin" / "java.exe").exists():
                    discovered.append(child)

        # Prefer higher versions first by directory name sorting (simple heuristic).
        for candidate in sorted(discovered, key=lambda p: p.name, reverse=True):
            env = build_env_with_java_home(candidate)
            major = java_major_from_cmd(["java", "-version"], env=env)
            if major is not None and major >= 11:
                return env

    raise RuntimeError(
        "No Java 11+ runtime found for Joern. Install JDK 11 or newer, "
        "or pass --java-home <path_to_jdk>."
    )


def resolve_executable(value: str, tool_name: str) -> str:
    # 1) Explicit path from CLI
    explicit = Path(value)
    if explicit.exists():
        return str(explicit)

    # 2) PATH lookup as-is
    found = shutil.which(value)
    if found:
        if os.name != "nt":
            return found
        if Path(found).suffix.lower() in {".bat", ".cmd", ".exe"}:
            return found

    # 3) Windows extension fallback for common Joern launcher names
    if os.name == "nt" and not Path(value).suffix:
        for suffix in (".bat", ".cmd", ".exe"):
            found = shutil.which(value + suffix)
            if found:
                return found

    # 4) JOERN_HOME fallback
    joern_home = os.environ.get("JOERN_HOME")
    if joern_home:
        for suffix in candidate_suffixes():
            candidate = Path(joern_home) / "bin" / f"{tool_name}{suffix}"
            if candidate.exists():
                return str(candidate)

    # 5) Project-local fallback: GRACE-java/tools/joern/**/bin/joern-*
    project_joern_root = WORK_DIR.parent / "tools" / "joern"
    if project_joern_root.exists():
        for suffix in candidate_suffixes():
            direct = project_joern_root / "joern-cli" / "bin" / f"{tool_name}{suffix}"
            if direct.exists():
                return str(direct)

        candidates = sorted(project_joern_root.rglob(f"{tool_name}*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.parent.name != "bin":
                continue
            suffix = candidate.suffix.lower()
            if os.name == "nt":
                if suffix in {".bat", ".cmd", ".exe"}:
                    return str(candidate)
            elif suffix in {"", ".sh", ".bat", ".cmd", ".exe"}:
                return str(candidate)

    raise FileNotFoundError(
        "Cannot find Joern executable for "
        f"'{tool_name}'. Provide full paths via --joern-parse/--joern-export, "
        "or add Joern bin directory to PATH, set JOERN_HOME, "
        "or place Joern under GRACE-java/tools/joern."
    )


def first_dot_file(directory: Path) -> str | None:
    dots = sorted(directory.rglob("*.dot"))
    return str(dots[0]) if dots else None


def main() -> None:
    args = parse_args()
    joern_env = resolve_java_env(args.java_home)
    joern_parse = resolve_executable(args.joern_parse, "joern-parse")
    joern_export = resolve_executable(args.joern_export, "joern-export")
    output_path = Path(args.output).resolve()

    print(f"[Joern] Java: {java_version_line(joern_env)}")
    print(f"[Joern] Java executable: {java_executable_path(joern_env)}")
    print(f"[Joern] joern-parse: {joern_parse}")
    print(f"[Joern] joern-export: {joern_export}")

    rows = load_jsonl(args.input)
    out_rows = []
    base_out = ensure_dir(output_path.parent / (output_path.stem + "_joern"))

    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", idx))
        # Keep output directory unique even when long sample_id values are truncated by slugify.
        sample_slug = f"{slugify(sample_id, max_len=64)}_{idx}"
        workdir = Path(tempfile.mkdtemp(prefix=f"joern_{sample_slug}_"))
        srcdir = ensure_dir(workdir / "src")
        class_name = f"Wrap_{sample_slug}"
        java_path = srcdir / f"{class_name}.java"
        java_path.write_text(wrap_java_method(row["code"], class_name), encoding="utf-8")

        cpg_path = workdir / "cpg.bin.zip"
        # joern-export requires --out directory to not exist beforehand.
        ast_out = workdir / "ast"
        cfg_out = workdir / "cfg"
        pdg_out = workdir / "pdg"

        status = "ok"
        error = None
        try:
            run_cmd([joern_parse, str(srcdir), "--language", "JAVASRC", "--output", str(cpg_path)], env=joern_env)
            run_cmd([joern_export, str(cpg_path), "--repr", "ast", "--format", "dot", "--out", str(ast_out)], env=joern_env)
            run_cmd([joern_export, str(cpg_path), "--repr", "cfg", "--format", "dot", "--out", str(cfg_out)], env=joern_env)
            run_cmd([joern_export, str(cpg_path), "--repr", "pdg", "--format", "dot", "--out", str(pdg_out)], env=joern_env)
        except Exception as e:  # noqa: BLE001
            status = "failed"
            error = str(e)

        sample_out = ensure_dir(base_out / sample_slug)
        if status == "ok":
            ast_file = first_dot_file(ast_out)
            cfg_file = first_dot_file(cfg_out)
            pdg_file = first_dot_file(pdg_out)
            if not any([ast_file, cfg_file, pdg_file]):
                status = "failed"
                error = (
                    "Joern completed but produced no AST/CFG/PDG dot files. "
                    "Input snippet may be incomplete/unparsable when wrapped as a Java class."
                )
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
