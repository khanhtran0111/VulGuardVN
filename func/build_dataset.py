import re
import json
import random
import pandas as pd
from pathlib import Path
from git import Repo
from unidiff import PatchSet
from tree_sitter_language_pack import get_parser

random.seed(0)
JAVA_PARSER = get_parser("java")

def ts_methods(source_code: str):
    if not source_code:
        return []
    
    tree = JAVA_PARSER.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    methods = []

    def traverse(node):
        if node.type in ("method_declaration", "constructor_declaration"):
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            name_node = node.child_by_field_name("name")
            method_name = source_code[name_node.start_byte:name_node.end_byte] if name_node else "unknown"
            code_text = source_code[node.start_byte:node.end_byte]

            methods.append({
                "name": method_name,
                "start_line": start_line,
                "end_line": end_line,
                "code": code_text
            })
        
        for child in node.children:
            traverse(child)
    
    traverse(root_node)
    return methods

def read_csv_flexible(path: str) -> pd.DataFrame:
    print(f"Reading CSV from: {path}")
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", engine="python")
            if df.shape[1] >= 2:
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception:
            pass
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_commit_from_human_patch_url(url: str):
    url = (url or "").strip()
    m = re.search(r"github\.com/([^/]+/[^/]+)/commit/([0-9a-fA-F]{7,40})", url)
    if not m:
        return None, None
    slug, sha = m.group(1), m.group(2)
    repo_url = f"https://github.com/{slug}.git"
    return repo_url, sha

def changed_line_ranges_both(unified_diff_text: str):
    patch = PatchSet(unified_diff_text.splitlines(True))
    old_ranges, new_ranges = [], []
    for f in patch:
        for h in f:
            if h.source_length and h.source_length > 0:
                old_ranges.append((h.source_start, h.source_start + h.source_length - 1))
            if h.target_length and h.target_length > 0:
                new_ranges.append((h.target_start, h.target_start + h.target_length - 1))
    return old_ranges, new_ranges

def method_hits(method, ranges):
    for a, b in ranges:
        if not (method["end_line"] < a or method["start_line"] > b):
            return True
    return False

def git_show(repo: Repo, commit: str, filepath: str):
    try:
        return repo.git.show(f"{commit}:{filepath}")
    except Exception:
        return None

def main(vul4j_csv: str, out_jsonl: str, cache_dir: str = "repo_cache", neg_ratio: int = 5, limit: int = 0):
    if not Path(vul4j_csv).exists():
        print(f"Error: File not found: {vul4j_csv}")
        return

    df = read_csv_flexible(vul4j_csv)
    df = df.fillna("")
    
    print(f"Loaded {len(df)} rows from CSV.")

    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)
    n_have_keys = 0
    n_parsed_url = 0
    n_wrote = 0
    n_skip_no_java_files = 0
    n_skip_no_pos = 0

    with out.open("w", encoding="utf-8") as w:
        for i, row in enumerate(df.to_dict(orient="records")):
            if limit and i >= limit:
                break
            
            if (i + 1) % 10 == 0:
                print(f"Processing row {i+1}/{n_rows}...")

            vul_id = (row.get("vul_id") or row.get("VUL_ID") or row.get("id") or "").strip()
            human_patch_url = (row.get("human_patch") or row.get("human_patch_url") or row.get("HUMAN_PATCH_URL") or "").strip()

            if not vul_id or not human_patch_url:
                continue
            n_have_keys += 1

            repo_url, fix_sha = parse_commit_from_human_patch_url(human_patch_url)
            if not repo_url or not fix_sha:
                continue
            n_parsed_url += 1

            repo_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", repo_url)
            local_repo_path = cache / repo_slug

            try:
                if not local_repo_path.exists():
                    print(f"  Cloning {repo_url}...")
                    Repo.clone_from(repo_url, local_repo_path)
                repo = Repo(local_repo_path) 
            except Exception as e:
                print(f"  Error accessing repo {repo_url}: {e}")
                continue

            try:
                parent_sha = repo.git.rev_parse(f"{fix_sha}^")
            except Exception:
                print(f"  Commit {fix_sha} not found in {repo_slug}")
                continue

            files = repo.git.diff("--name-only", parent_sha, fix_sha, "--", "*.java").splitlines()
            if not files:
                n_skip_no_java_files += 1
                continue

            for fp in files:
                old_code = git_show(repo, parent_sha, fp)
                new_code = git_show(repo, fix_sha, fp)
                if old_code is None or new_code is None:
                    continue

                udiff = repo.git.diff(parent_sha, fix_sha, "--unified=0", "--", fp)
                old_ranges, new_ranges = changed_line_ranges_both(udiff)

                old_methods = ts_methods(old_code)
                new_methods = ts_methods(new_code)

                pos_old = [m for m in old_methods if method_hits(m, old_ranges)]

                pos = pos_old
                if not pos:
                    hit_new = [m for m in new_methods if method_hits(m, new_ranges)]
                    hit_names = {m["name"] for m in hit_new}
                    pos = [m for m in old_methods if m["name"] in hit_names]

                if not pos:
                    n_skip_no_pos += 1
                    continue

                new_by_name = {}
                for m in new_methods:
                    new_by_name.setdefault(m["name"], []).append(m)

                for m in pos:
                    w.write(json.dumps({
                        "vul_id": vul_id,
                        "file": fp,
                        "method": m["name"],
                        "version": "vulnerable",
                        "label": 1,
                        "code": m["code"],
                    }, ensure_ascii=False) + "\n")
                    n_wrote += 1

                    cands = new_by_name.get(m["name"], [])
                    if cands:
                        nm = max(cands, key=lambda x: len(x["code"]))
                        w.write(json.dumps({
                            "vul_id": vul_id,
                            "file": fp,
                            "method": nm["name"],
                            "version": "fixed",
                            "label": 0,
                            "code": nm["code"],
                        }, ensure_ascii=False) + "\n")
                        n_wrote += 1

                neg_pool = [m for m in old_methods if m not in pos]
                random.shuffle(neg_pool)
                extra_negs = neg_pool[: max(1, neg_ratio * len(pos))]
                for m in extra_negs:
                    w.write(json.dumps({
                        "vul_id": vul_id,
                        "file": fp,
                        "method": m["name"],
                        "version": "vulnerable_other",
                        "label": 0,
                        "code": m["code"],
                    }, ensure_ascii=False) + "\n")
                    n_wrote += 1

    print("==== Summary ====")
    print("Total rows:", n_rows)
    print("Rows with keys:", n_have_keys)
    print("Parsed URLs:", n_parsed_url)
    print("Skipped (no .java):", n_skip_no_java_files)
    print("Skipped (no methods):", n_skip_no_pos)
    print("Written lines:", n_wrote)

if __name__ == "__main__":
    INPUT_CSV = "/home/khanhtran/Documents/NCKH/vul4j/dataset/vul4j_dataset.csv"
    OUTPUT_JSONL = "/home/khanhtran/Documents/NCKH/VulGuardVN/dataoutput_dataset.jsonl"
    
    main(
        vul4j_csv=INPUT_CSV,
        out_jsonl=OUTPUT_JSONL,
        limit=0
    )