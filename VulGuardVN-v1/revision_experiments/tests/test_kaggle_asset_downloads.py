from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REVISION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_DIR))

from kaggle_preflight import (  # noqa: E402
    prepare_required_assets,
    resolve_or_download_devign,
    resolve_or_download_hf_model,
    validate_devign_file,
)


DEVIGN_BYTES = json.dumps([
    {"func": "int safe(void) { return 0; }", "target": 0},
    {"func": "int vulnerable(char *s) { char b[4]; strcpy(b, s); }", "target": 1},
]).encode("utf-8")


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.content


def write_devign(path: Path, content: bytes = DEVIGN_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def write_model(path: Path, *, sharded: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    if sharded:
        (path / "model-00001-of-00002.safetensors").write_bytes(b"weights")
        (path / "model-00002-of-00002.safetensors").write_bytes(b"weights")
        (path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    else:
        (path / "model.safetensors").write_bytes(b"weights")
    return path


class KaggleAssetDownloadTests(unittest.TestCase):
    def test_devign_local_override_has_priority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo = root / "repo"; assets = root / "assets"
            override = write_devign(root / "mounted" / "function.json")
            write_devign(repo / "GRACE-improve" / "data" / "function.json", json.dumps([{"func": "repo", "target": 0}]).encode())
            resolved, status, usable = resolve_or_download_devign(
                repository_root=repo, asset_root=assets, source=override, urls=(), auto_download=False,
            )
            self.assertEqual(status, "mounted")
            self.assertEqual(usable, 2)
            self.assertEqual(resolved.read_bytes(), DEVIGN_BYTES)
            self.assertEqual((repo / "GRACE-improve" / "data" / "function.json").read_bytes(), DEVIGN_BYTES)

    def test_devign_cache_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); assets = root / "assets"
            cached = write_devign(assets / "datasets" / "devign" / "function.json")
            resolved, status, usable = resolve_or_download_devign(
                repository_root=root / "repo", asset_root=assets, source=None,
                urls=("https://example.invalid/function.json",), auto_download=True,
                requests_get=lambda *args, **kwargs: self.fail("network must not be called"),
            )
            self.assertEqual((resolved, status, usable), (cached, "cached", 2))

    def test_google_drive_url_is_delegated_to_gdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); calls = []

            def fake_gdown(**kwargs):
                calls.append(kwargs)
                write_devign(Path(kwargs["output"]))
                return kwargs["output"]

            _, status, _ = resolve_or_download_devign(
                repository_root=root / "repo", asset_root=root / "assets", source=None,
                urls=("https://drive.google.com/file/d/fixture/view",), auto_download=True,
                gdown_download=fake_gdown, sleep=lambda _: None,
            )
            self.assertEqual(status, "downloaded")
            self.assertEqual(calls[0]["url"], "https://drive.google.com/file/d/fixture/view")
            self.assertTrue(calls[0]["fuzzy"])

    def test_raw_json_url_downloads_successfully(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); calls = []

            def fake_get(url, **kwargs):
                calls.append((url, kwargs))
                return FakeResponse(DEVIGN_BYTES)

            resolved, status, usable = resolve_or_download_devign(
                repository_root=root / "repo", asset_root=root / "assets", source=None,
                urls=("https://raw.example/function.json",), auto_download=True,
                requests_get=fake_get, sleep=lambda _: None,
            )
            self.assertEqual(status, "downloaded")
            self.assertEqual(usable, 2)
            self.assertEqual(validate_devign_file(resolved), 2)
            self.assertEqual(calls[0][1]["timeout"], 120)

    def test_html_response_is_rejected_then_next_url_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); counts = {"html": 0, "json": 0}

            def fake_get(url, **kwargs):
                del kwargs
                key = "html" if "html" in url else "json"
                counts[key] += 1
                return FakeResponse(b"<!doctype html><html>Google Drive warning</html>" if key == "html" else DEVIGN_BYTES)

            resolved, status, _ = resolve_or_download_devign(
                repository_root=root / "repo", asset_root=root / "assets", source=None,
                urls=("https://example/html", "https://example/json"), auto_download=True,
                requests_get=fake_get, sleep=lambda _: None,
            )
            self.assertEqual(status, "downloaded")
            self.assertEqual(validate_devign_file(resolved), 2)
            self.assertEqual(counts, {"html": 3, "json": 1})
            self.assertFalse(any((root / "assets" / "downloads").glob("*.part")))

    def test_invalid_json_is_deleted_and_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Unable to download a valid Devign"):
                resolve_or_download_devign(
                    repository_root=root / "repo", asset_root=root / "assets", source=None,
                    urls=("https://example/invalid",), auto_download=True,
                    requests_get=lambda *args, **kwargs: FakeResponse(b"not valid json data"),
                    sleep=lambda _: None,
                )
            self.assertFalse((root / "assets" / "datasets" / "devign" / "function.json").exists())
            self.assertFalse(any((root / "assets" / "downloads").glob("*.part")))

    def test_unixcoder_is_downloaded_when_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); calls = []

            def fake_snapshot(**kwargs):
                calls.append(kwargs)
                write_model(Path(kwargs["local_dir"]))

            resolved, status = resolve_or_download_hf_model(
                asset_root=root / "assets", model_id="microsoft/unixcoder-base-nine",
                source_dir=None, auto_download=True, snapshot_download_fn=fake_snapshot,
            )
            self.assertEqual(status, "downloaded")
            self.assertEqual(resolved.name, "microsoft--unixcoder-base-nine")
            self.assertFalse(calls[0]["local_dir_use_symlinks"])

    def test_qwen_is_downloaded_for_full_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); mounted_devign = write_devign(root / "input" / "function.json"); calls = []

            def fake_snapshot(**kwargs):
                calls.append(kwargs["repo_id"])
                write_model(Path(kwargs["local_dir"]), sharded="Qwen" in kwargs["repo_id"])

            prepared = prepare_required_assets(
                repository_root=root / "repo", asset_root=root / "assets",
                devign_source=mounted_devign, devign_urls=(), devign_archive_member="",
                retrieval_model_source=None, retrieval_model_id="microsoft/unixcoder-base-nine",
                local_llm_source=None, local_llm_model_id="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
                hf_token="fixture-token", auto_download_dataset=True, auto_download_models=True,
                require_llm=True, snapshot_download_fn=fake_snapshot,
            )
            self.assertIsNotNone(prepared.local_llm_model_dir)
            self.assertEqual(prepared.local_llm_model_source, "downloaded")
            self.assertEqual(calls, ["microsoft/unixcoder-base-nine", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"])

    def test_qwen_is_skipped_for_smoke_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); mounted_devign = write_devign(root / "input" / "function.json"); calls = []

            def fake_snapshot(**kwargs):
                calls.append(kwargs["repo_id"])
                write_model(Path(kwargs["local_dir"]))

            prepared = prepare_required_assets(
                repository_root=root / "repo", asset_root=root / "assets",
                devign_source=mounted_devign, devign_urls=(), devign_archive_member="",
                retrieval_model_source=None, retrieval_model_id="microsoft/unixcoder-base-nine",
                local_llm_source=None, local_llm_model_id="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
                hf_token="", auto_download_dataset=True, auto_download_models=True,
                require_llm=False, snapshot_download_fn=fake_snapshot,
            )
            self.assertIsNone(prepared.local_llm_model_dir)
            self.assertEqual(calls, ["microsoft/unixcoder-base-nine"])

    def test_model_cache_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); target = write_model(root / "assets" / "models" / "microsoft--unixcoder-base-nine")
            resolved, status = resolve_or_download_hf_model(
                asset_root=root / "assets", model_id="microsoft/unixcoder-base-nine",
                source_dir=None, auto_download=True,
                snapshot_download_fn=lambda **kwargs: self.fail("snapshot download must not run"),
            )
            self.assertEqual((resolved, status), (target, "cached"))

    def test_full_run_all_preflight_succeeds_with_mocked_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); snapshot_calls = []

            def fake_snapshot(**kwargs):
                snapshot_calls.append(kwargs["repo_id"])
                write_model(Path(kwargs["local_dir"]), sharded="Qwen" in kwargs["repo_id"])

            assets = prepare_required_assets(
                repository_root=root / "repo", asset_root=root / "working" / "vulguard-assets",
                devign_source=None, devign_urls=("https://raw.example/function.json",), devign_archive_member="",
                retrieval_model_source=None, retrieval_model_id="microsoft/unixcoder-base-nine",
                local_llm_source=None, local_llm_model_id="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
                hf_token="", auto_download_dataset=True, auto_download_models=True, require_llm=True,
                requests_get=lambda *args, **kwargs: FakeResponse(DEVIGN_BYTES),
                snapshot_download_fn=fake_snapshot, sleep=lambda _: None,
            )
            self.assertEqual(validate_devign_file(root / "repo" / "GRACE-improve" / "data" / "function.json"), 2)
            self.assertEqual(os.environ["GRACE_RETRIEVAL_MODEL_DIR"], str(assets.retrieval_model_dir))
            self.assertEqual(os.environ["GRACE_LOCAL_MODEL_DIR"], str(assets.local_llm_model_dir))
            self.assertEqual(len(snapshot_calls), 2)


if __name__ == "__main__":
    unittest.main()
