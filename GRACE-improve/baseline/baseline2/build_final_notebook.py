import json
import textwrap
from pathlib import Path


BASELINE2_DIR = Path(__file__).resolve().parent
FINAL_NOTEBOOK_PATH = BASELINE2_DIR / "FINAL.ipynb"

MODULE_FILES = [
    "common.py",
    "datasets.py",
    "graphs.py",
    "metrics.py",
    "localizer.py",
    "retrieval.py",
    "hybrid_prefilter.py",
    "local_llm_client.py",
    "evaluate_predictions.py",
]

STEP_FILES = [
    "00_verify_assets.py",
    "01_prepare_datasets.py",
    "02_create_splits.py",
    "03_build_feature_store.py",
    "04_train_hybrid_prefilter.py",
    "05_calibrate_budget_controller.py",
    "06_build_demo_bank.py",
    "07_run_grace_hybrid.py",
    "08_evaluate_predictions.py",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def to_source_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_lines(textwrap.dedent(text).lstrip("\n")),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(textwrap.dedent(text).lstrip("\n")),
    }


def writefile_cell(relative_path: str, source: str) -> dict:
    payload = f"%%writefile {relative_path}\n{source}"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(payload),
    }


def load_notebook_metadata() -> dict:
    if FINAL_NOTEBOOK_PATH.exists():
        return json.loads(read_text(FINAL_NOTEBOOK_PATH)).get("metadata", {})
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }


def build_config_cell() -> str:
    return """
    from pathlib import Path

    # =========================
    # Preconfigured for Kaggle
    # No edits needed for Devign
    # =========================

    DATASET_NAME = 'devign'  # 'devign' | 'bigvul' | 'reveal'

    # Kaggle paths
    KAGGLE_INPUT_ROOT = Path('/kaggle/input')
    WORKING_ROOT = Path('/kaggle/working/vulguardvn-final')

    # Dataset sources from Kaggle Input.
    # Set only the paths needed for the dataset you want to run.
    DEVIGN_SOURCE_PATH = None
    BIGVUL_SOURCE_PATH = None
    REVEAL_SOURCE_DIR = None

    # Optional direct download config for datasets when Kaggle Input is not mounted.
    AUTO_DOWNLOAD_DATASET_IF_MISSING = True
    DEVIGN_DOWNLOAD_URL = 'https://raw.githubusercontent.com/madlag/CodeXGLUE/main/Code-Code/Defect-detection/dataset/function.json'
    DEVIGN_ARCHIVE_MEMBER = ''
    BIGVUL_DOWNLOAD_URL = ''
    BIGVUL_ARCHIVE_MEMBER = ''

    # Optional local model folders uploaded as Kaggle datasets.
    RETRIEVAL_MODEL_SOURCE_DIR = None
    LOCAL_LLM_SOURCE_DIR = None

    # Optional Hugging Face token. Public defaults work without it, but a token can make downloads more reliable.
    HF_TOKEN = ''

    # Staging toggles
    USE_SYMLINKS_WHEN_POSSIBLE = True
    COPY_MODELS_INSTEAD_OF_LINK = False
    RESET_WORKING_ROOT = False
    AUTO_DOWNLOAD_MISSING_MODELS = True
    AUTO_DOWNLOAD_REVEAL_IF_MISSING = False
    RESTORE_PREVIOUS_KAGGLE_OUTPUT = True
    PREVIOUS_OUTPUT_SOURCE_DIR = None

    # Core pipeline options
    GRAPH_BACKEND = 'auto'            # 'auto' | 'heuristic' | 'joern'
    RETRIEVAL_MODEL_ID = 'microsoft/unixcoder-base-nine'
    LOCAL_LLM_MODEL_ID = 'unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit'
    PREFILTER_MODEL_NAME = 'hybrid_multiview_prefilter'

    # LLM / inference options
    # These defaults are tuned for better F1 with a controlled LLM budget on Kaggle T4.
    LOAD_IN_4BIT = True
    CALL_LLM_FOR_INSPECT = True
    CALL_LLM_FOR_HIGH = False
    MAX_TEST_SAMPLES = None
    TEST_CHUNK_SIZE = 64
    RUN_ALL_TEST_CHUNKS_IN_ONE_RUN = True
    INSPECT_DEMOS = 2
    HIGH_RISK_DEMOS = 1
    MAX_NEW_TOKENS = 96
    DEMO_CHAR_LIMIT = 800
    PROMPT_CODE_CHAR_LIMIT = 1800
    PROMPT_TOP_LINES_LIMIT = 3
    PROMPT_TOP_LINE_CHAR_LIMIT = 120
    PROMPT_SLICES_CHAR_LIMIT = 900
    PROMPT_NODE_INFO_CHAR_LIMIT = 900
    PROMPT_EDGE_INFO_CHAR_LIMIT = 900
    RESUME_RUN = True

    # Speed / training options
    FEATURE_BATCH_SIZE = 16
    FEATURE_PROGRESS_EVERY = 256
    BUILD_PROGRESS_EVERY = 250
    PREFILTER_BATCH_SIZE = 128
    PREFILTER_EPOCHS = 10
    PREFILTER_LEARNING_RATE = 7e-4
    TARGET_RECALL = 0.995
    DIRECT_ACCEPT_MIN_PROBABILITY = 0.20
    HIGH_RISK_THRESHOLD_STRATEGY = 'f1'
    HIGH_RISK_TARGET_PRECISION = 0.70

    # Optional feature limits for debugging
    GRACE_FEATURE_LIMIT = None
    GRACE_FEATURE_LIMIT_TRAIN = None
    GRACE_FEATURE_LIMIT_VAL = None
    GRACE_FEATURE_LIMIT_TEST = None
    """


def build_install_cell() -> str:
    return """
    import importlib.util
    import subprocess
    import sys

    required_pip_packages = {
        'dotenv': 'python-dotenv',
        'joblib': 'joblib',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'pyarrow': 'pyarrow',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'huggingface_hub': 'huggingface_hub',
        'sentencepiece': 'sentencepiece',
        'bitsandbytes': 'bitsandbytes',
        'gdown': 'gdown',
        'requests': 'requests',
    }

    missing = [package for module_name, package in required_pip_packages.items() if importlib.util.find_spec(module_name) is None]
    if missing:
        print('Installing missing packages:', missing)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
    else:
        print('All required pip packages are already available.')

    for heavy_module in ['torch', 'tensorflow']:
        if importlib.util.find_spec(heavy_module) is None:
            raise RuntimeError(f'Missing required runtime package on Kaggle image: {heavy_module}')
    """


def build_setup_cell() -> str:
    return """
    import gzip
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    import urllib.parse
    import zipfile
    from pathlib import Path

    import requests


    def as_path(value):
        if value is None or value == '':
            return None
        if isinstance(value, Path):
            return value.expanduser().resolve()
        return Path(str(value)).expanduser().resolve()


    def iter_dirs(root, max_depth=3):
        root = as_path(root)
        if root is None or not root.exists():
            return
        queue = [(root, 0)]
        seen = set()
        while queue:
            current, depth = queue.pop(0)
            key = str(current)
            if key in seen:
                continue
            seen.add(key)
            yield current
            if depth >= max_depth:
                continue
            try:
                children = sorted([child for child in current.iterdir() if child.is_dir()])
            except Exception:
                continue
            for child in children:
                queue.append((child, depth + 1))


    def find_existing_file(candidates):
        for candidate in candidates:
            candidate = as_path(candidate)
            if candidate is not None and candidate.is_file():
                return candidate
        return None


    def search_for_file(filename, search_roots, max_depth=4):
        for search_root in search_roots:
            search_root = as_path(search_root)
            if search_root is None or not search_root.exists():
                continue
            direct = search_root / filename
            if direct.is_file():
                return direct.resolve()
            for candidate in iter_dirs(search_root, max_depth=max_depth):
                path = candidate / filename
                if path.is_file():
                    return path.resolve()
        return None


    def search_for_dirname(dirname, search_roots, max_depth=4):
        for search_root in search_roots:
            search_root = as_path(search_root)
            if search_root is None or not search_root.exists():
                continue
            direct = search_root / dirname
            if direct.is_dir():
                return direct.resolve()
            for candidate in iter_dirs(search_root, max_depth=max_depth):
                if candidate.name == dirname and candidate.is_dir():
                    return candidate.resolve()
        return None


    def looks_like_model_dir(path):
        path = as_path(path)
        if path is None or not path.is_dir():
            return False
        has_config = (path / 'config.json').exists()
        has_weights = any(path.glob('*.safetensors')) or any(path.glob('pytorch_model*.bin'))
        return has_config and has_weights


    def coerce_model_dir(explicit_path, expected_dir_name):
        explicit_path = as_path(explicit_path)
        if explicit_path is None:
            return None
        if looks_like_model_dir(explicit_path):
            return explicit_path
        nested = explicit_path / expected_dir_name
        if looks_like_model_dir(nested):
            return nested.resolve()
        found = search_for_dirname(expected_dir_name, [explicit_path], max_depth=3)
        if found is not None and looks_like_model_dir(found):
            return found
        return explicit_path if explicit_path.exists() else None


    def looks_like_reveal_processed_dir(path):
        path = as_path(path)
        if path is None or not path.is_dir():
            return False
        required = ['train.jsonl', 'val.jsonl', 'test.jsonl']
        return all((path / name).is_file() for name in required)


    def looks_like_reveal_parquet_dir(path):
        path = as_path(path)
        if path is None or not path.is_dir():
            return False
        required = [
            'train-00000-of-00001.parquet',
            'validation-00000-of-00001.parquet',
            'test-00000-of-00001.parquet',
        ]
        return all((path / name).is_file() for name in required)


    def coerce_reveal_dir(explicit_path):
        explicit_path = as_path(explicit_path)
        if explicit_path is None:
            return None
        if looks_like_reveal_processed_dir(explicit_path) or looks_like_reveal_parquet_dir(explicit_path):
            return explicit_path
        for child_name in ['reveal', 'reveal_raw', 'reveal_ready', 'ReVeal', 'Reveal']:
            child = explicit_path / child_name
            if looks_like_reveal_processed_dir(child) or looks_like_reveal_parquet_dir(child) or child.is_dir():
                return child.resolve()
        return explicit_path if explicit_path.exists() else None


    def normalize_previous_output_root(path):
        path = as_path(path)
        if path is None or not path.exists():
            return None
        if (path / 'GRACE-improve' / 'baseline' / 'baseline2' / 'artifacts').exists():
            return path.resolve()
        if path.name == 'GRACE-improve' and (path / 'baseline' / 'baseline2' / 'artifacts').exists():
            return path.parent.resolve()
        return None


    def score_previous_output_root(root):
        code_root = root / 'GRACE-improve'
        baseline2_artifacts = code_root / 'baseline' / 'baseline2' / 'artifacts'
        shared_artifacts = code_root / 'baseline' / 'artifacts'
        dataset_predictions = baseline2_artifacts / 'predictions' / DATASET_NAME / 'grace_hybrid_predictions.jsonl'
        dataset_run_state = baseline2_artifacts / 'predictions' / DATASET_NAME / 'grace_hybrid_run_state.json'
        score = 0
        if baseline2_artifacts.exists():
            score += 10
        if shared_artifacts.exists():
            score += 5
        if dataset_predictions.exists():
            score += 20
        if dataset_run_state.exists():
            score += 20
        return score


    def resolve_previous_output_root(explicit_path=None):
        explicit_root = normalize_previous_output_root(explicit_path)
        if explicit_root is not None:
            return explicit_root
        candidates = []
        if KAGGLE_INPUT_ROOT.exists():
            direct_root = normalize_previous_output_root(KAGGLE_INPUT_ROOT)
            if direct_root is not None:
                candidates.append(direct_root)
            for candidate in iter_dirs(KAGGLE_INPUT_ROOT, max_depth=5):
                normalized = normalize_previous_output_root(candidate)
                if normalized is not None:
                    candidates.append(normalized)
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        if not unique_candidates:
            return None
        unique_candidates.sort(key=lambda path: (score_previous_output_root(path), len(str(path))), reverse=True)
        return unique_candidates[0]


    def restore_previous_artifacts(previous_output_root):
        previous_output_root = normalize_previous_output_root(previous_output_root)
        if previous_output_root is None:
            print('No previous Kaggle output artifacts detected; starting from a clean workspace state.')
            return None
        previous_code_root = previous_output_root / 'GRACE-improve'
        restore_pairs = [
            (
                previous_code_root / 'baseline' / 'artifacts',
                WORKING_CODE_ROOT / 'baseline' / 'artifacts',
                'shared baseline artifacts',
            ),
            (
                previous_code_root / 'baseline' / 'baseline2' / 'artifacts',
                WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts',
                'baseline2 artifacts',
            ),
        ]
        restored = []
        for src, dst, label in restore_pairs:
            if src.exists():
                print(f'Restoring {label}: {src} -> {dst}')
                shutil.copytree(src, dst, dirs_exist_ok=True)
                restored.append({'label': label, 'source': str(src), 'destination': str(dst)})
        if not restored:
            print(f'Previous output root found but no restorable artifacts were detected: {previous_output_root}')
            return None
        print(json.dumps({'restored_from': str(previous_output_root), 'restored': restored}, indent=2))
        return previous_output_root


    def infer_filename_from_url(url, fallback_name):
        parsed = urllib.parse.urlparse(url)
        name = Path(urllib.parse.unquote(parsed.path)).name
        return name or fallback_name


    def download_file(url, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        headers = {}
        if HF_TOKEN and 'huggingface.co' in url:
            headers['Authorization'] = f'Bearer {HF_TOKEN.strip()}'
        with requests.get(url, stream=True, timeout=120, allow_redirects=True, headers=headers) as response:
            response.raise_for_status()
            with destination.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        return destination


    def extract_downloaded_asset(archive_path, expected_filename, archive_member=''):
        archive_path = Path(archive_path)
        extract_dir = archive_path.parent / f'{archive_path.name}.extracted'
        extract_dir.mkdir(parents=True, exist_ok=True)
        archive_member = (archive_member or '').strip()

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                if archive_member:
                    zf.extract(archive_member, extract_dir)
                    candidate = extract_dir / archive_member
                    if candidate.exists():
                        return candidate.resolve()
                else:
                    zf.extractall(extract_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tf:
                if archive_member:
                    tf.extract(archive_member, extract_dir)
                    candidate = extract_dir / archive_member
                    if candidate.exists():
                        return candidate.resolve()
                else:
                    tf.extractall(extract_dir)
        elif archive_path.suffix.lower() == '.gz' and not archive_path.name.endswith('.tar.gz'):
            target = extract_dir / (archive_member or expected_filename)
            with gzip.open(archive_path, 'rb') as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if target.exists():
                return target.resolve()
        else:
            return None

        found = search_for_file(expected_filename, [extract_dir], max_depth=8)
        return found.resolve() if found is not None else None


    def ensure_dataset_file(dataset_name, source_path, download_url, expected_filename, archive_member=''):
        source_path = as_path(source_path)
        if source_path is not None and source_path.exists():
            return source_path, None
        if not AUTO_DOWNLOAD_DATASET_IF_MISSING:
            return None, None
        if not download_url:
            raise FileNotFoundError(
                f'{dataset_name} source file is missing. Set the *_SOURCE_PATH or provide {dataset_name.upper()}_DOWNLOAD_URL.'
            )
        dataset_download_dir = WORKING_ROOT / '_downloads' / dataset_name
        dataset_download_dir.mkdir(parents=True, exist_ok=True)
        raw_name = infer_filename_from_url(download_url, expected_filename)
        raw_path = dataset_download_dir / raw_name
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            print(f'Downloading {dataset_name} dataset from {download_url}')
            download_file(download_url, raw_path)
        else:
            print(f'Using cached downloaded asset for {dataset_name}: {raw_path}')

        if raw_path.name == expected_filename:
            return raw_path.resolve(), {'mode': 'download', 'source': download_url, 'path': str(raw_path.resolve())}

        extracted = extract_downloaded_asset(raw_path, expected_filename, archive_member=archive_member)
        if extracted is not None and extracted.exists():
            return extracted.resolve(), {'mode': 'download+extract', 'source': download_url, 'path': str(extracted.resolve())}

        located = search_for_file(expected_filename, [dataset_download_dir], max_depth=8)
        if located is not None:
            return located.resolve(), {'mode': 'download+locate', 'source': download_url, 'path': str(located.resolve())}

        raise FileNotFoundError(
            f'Could not locate {expected_filename} after downloading {dataset_name} from {download_url}. '
            'If the file is inside an archive, set the corresponding *_ARCHIVE_MEMBER in the config cell.'
        )


    def remove_existing_target(target):
        target = Path(target)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)


    def link_or_copy(source, target, prefer_symlink=True):
        source = as_path(source)
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            remove_existing_target(target)
        if prefer_symlink:
            try:
                os.symlink(str(source), str(target), target_is_directory=source.is_dir())
                return 'symlink'
            except Exception:
                pass
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
            return 'copytree'
        shutil.copy2(source, target)
        return 'copy'


    def set_or_clear_env(name, value):
        if value is None or value == '':
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


    KAGGLE_INPUT_ROOT = as_path(KAGGLE_INPUT_ROOT) or Path('/kaggle/input')
    WORKING_ROOT = as_path(WORKING_ROOT) or Path('/kaggle/working/vulguardvn-final')
    WORKING_CODE_ROOT = WORKING_ROOT / 'GRACE-improve'
    BASELINE_DIR = WORKING_CODE_ROOT / 'baseline'
    BASELINE2_DIR = BASELINE_DIR / 'baseline2'
    WORKING_DATA_DIR = WORKING_CODE_ROOT / 'data'
    WORKING_SHARED_ARTIFACTS_DIR = BASELINE_DIR / 'artifacts'
    WORKING_SHARED_MODELS_DIR = WORKING_SHARED_ARTIFACTS_DIR / 'models'
    WORKING_RETRIEVAL_TARGET_DIR = WORKING_SHARED_MODELS_DIR / 'retrieval' / RETRIEVAL_MODEL_ID.replace('/', '--')
    WORKING_LOCAL_LLM_TARGET_DIR = WORKING_SHARED_MODELS_DIR / 'local_llm' / LOCAL_LLM_MODEL_ID.replace('/', '--')

    if RESET_WORKING_ROOT and WORKING_ROOT.exists():
        print(f'Removing existing working root: {WORKING_ROOT}')
        shutil.rmtree(WORKING_ROOT)

    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    BASELINE2_DIR.mkdir(parents=True, exist_ok=True)
    WORKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORKING_SHARED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (WORKING_SHARED_MODELS_DIR / 'retrieval').mkdir(parents=True, exist_ok=True)
    (WORKING_SHARED_MODELS_DIR / 'local_llm').mkdir(parents=True, exist_ok=True)
    (BASELINE2_DIR / 'artifacts').mkdir(parents=True, exist_ok=True)

    RETRIEVAL_MODEL_DIRNAME = RETRIEVAL_MODEL_ID.replace('/', '--')
    LOCAL_LLM_DIRNAME = LOCAL_LLM_MODEL_ID.replace('/', '--')

    DEVIGN_SOURCE_PATH = find_existing_file([DEVIGN_SOURCE_PATH]) or search_for_file('function.json', [KAGGLE_INPUT_ROOT])
    BIGVUL_SOURCE_PATH = find_existing_file([BIGVUL_SOURCE_PATH]) or search_for_file('MSR_data_cleaned.csv', [KAGGLE_INPUT_ROOT])

    REVEAL_SOURCE_DIR = coerce_reveal_dir(REVEAL_SOURCE_DIR)
    if REVEAL_SOURCE_DIR is None:
        for candidate in iter_dirs(KAGGLE_INPUT_ROOT, max_depth=5):
            if looks_like_reveal_processed_dir(candidate) or looks_like_reveal_parquet_dir(candidate):
                REVEAL_SOURCE_DIR = candidate.resolve()
                break

    RETRIEVAL_MODEL_SOURCE_DIR = coerce_model_dir(RETRIEVAL_MODEL_SOURCE_DIR, RETRIEVAL_MODEL_DIRNAME)
    if RETRIEVAL_MODEL_SOURCE_DIR is None:
        found = search_for_dirname(RETRIEVAL_MODEL_DIRNAME, [KAGGLE_INPUT_ROOT], max_depth=5)
        if found is not None and looks_like_model_dir(found):
            RETRIEVAL_MODEL_SOURCE_DIR = found.resolve()

    LOCAL_LLM_SOURCE_DIR = coerce_model_dir(LOCAL_LLM_SOURCE_DIR, LOCAL_LLM_DIRNAME)
    if LOCAL_LLM_SOURCE_DIR is None:
        found = search_for_dirname(LOCAL_LLM_DIRNAME, [KAGGLE_INPUT_ROOT], max_depth=5)
        if found is not None and looks_like_model_dir(found):
            LOCAL_LLM_SOURCE_DIR = found.resolve()

    discovery = {
        'kaggle_input_root': str(KAGGLE_INPUT_ROOT),
        'working_root': str(WORKING_ROOT),
        'dataset_name': DATASET_NAME,
        'devign_source_path': str(DEVIGN_SOURCE_PATH) if DEVIGN_SOURCE_PATH else None,
        'bigvul_source_path': str(BIGVUL_SOURCE_PATH) if BIGVUL_SOURCE_PATH else None,
        'reveal_source_dir': str(REVEAL_SOURCE_DIR) if REVEAL_SOURCE_DIR else None,
        'retrieval_model_source_dir': str(RETRIEVAL_MODEL_SOURCE_DIR) if RETRIEVAL_MODEL_SOURCE_DIR else None,
        'local_llm_source_dir': str(LOCAL_LLM_SOURCE_DIR) if LOCAL_LLM_SOURCE_DIR else None,
    }
    print(json.dumps(discovery, indent=2))

    PREVIOUS_OUTPUT_ROOT = None
    if RESTORE_PREVIOUS_KAGGLE_OUTPUT:
        PREVIOUS_OUTPUT_ROOT = restore_previous_artifacts(resolve_previous_output_root(PREVIOUS_OUTPUT_SOURCE_DIR))
    else:
        print('RESTORE_PREVIOUS_KAGGLE_OUTPUT is disabled.')

    dataset_download_summary = {}
    if DATASET_NAME == 'devign':
        DEVIGN_SOURCE_PATH, devign_download_info = ensure_dataset_file(
            'devign',
            DEVIGN_SOURCE_PATH,
            DEVIGN_DOWNLOAD_URL,
            'function.json',
            archive_member=DEVIGN_ARCHIVE_MEMBER,
        )
        if devign_download_info is not None:
            dataset_download_summary['devign'] = devign_download_info
    elif DATASET_NAME == 'bigvul':
        BIGVUL_SOURCE_PATH, bigvul_download_info = ensure_dataset_file(
            'bigvul',
            BIGVUL_SOURCE_PATH,
            BIGVUL_DOWNLOAD_URL,
            'MSR_data_cleaned.csv',
            archive_member=BIGVUL_ARCHIVE_MEMBER,
        )
        if bigvul_download_info is not None:
            dataset_download_summary['bigvul'] = bigvul_download_info

    staging_summary = {}
    if DEVIGN_SOURCE_PATH is not None:
        target = WORKING_DATA_DIR / 'function.json'
        mode = link_or_copy(DEVIGN_SOURCE_PATH, target, prefer_symlink=USE_SYMLINKS_WHEN_POSSIBLE)
        staging_summary['devign'] = {'mode': mode, 'target': str(target)}
    if BIGVUL_SOURCE_PATH is not None:
        target = WORKING_DATA_DIR / 'MSR_data_cleaned.csv'
        mode = link_or_copy(BIGVUL_SOURCE_PATH, target, prefer_symlink=USE_SYMLINKS_WHEN_POSSIBLE)
        staging_summary['bigvul'] = {'mode': mode, 'target': str(target)}
    if REVEAL_SOURCE_DIR is not None:
        reveal_target_name = 'reveal' if looks_like_reveal_processed_dir(REVEAL_SOURCE_DIR) else 'reveal_raw'
        reveal_target = WORKING_DATA_DIR / reveal_target_name
        mode = link_or_copy(REVEAL_SOURCE_DIR, reveal_target, prefer_symlink=USE_SYMLINKS_WHEN_POSSIBLE)
        staging_summary['reveal'] = {'mode': mode, 'target': str(reveal_target)}
    if RETRIEVAL_MODEL_SOURCE_DIR is not None:
        mode = link_or_copy(
            RETRIEVAL_MODEL_SOURCE_DIR,
            WORKING_RETRIEVAL_TARGET_DIR,
            prefer_symlink=(USE_SYMLINKS_WHEN_POSSIBLE and not COPY_MODELS_INSTEAD_OF_LINK),
        )
        staging_summary['retrieval_model'] = {'mode': mode, 'target': str(WORKING_RETRIEVAL_TARGET_DIR)}
    if LOCAL_LLM_SOURCE_DIR is not None:
        mode = link_or_copy(
            LOCAL_LLM_SOURCE_DIR,
            WORKING_LOCAL_LLM_TARGET_DIR,
            prefer_symlink=(USE_SYMLINKS_WHEN_POSSIBLE and not COPY_MODELS_INSTEAD_OF_LINK),
        )
        staging_summary['local_llm'] = {'mode': mode, 'target': str(WORKING_LOCAL_LLM_TARGET_DIR)}

    if DATASET_NAME == 'devign' and not (WORKING_DATA_DIR / 'function.json').exists():
        raise FileNotFoundError('Devign file not found. Set DEVIGN_SOURCE_PATH or enable AUTO_DOWNLOAD_DATASET_IF_MISSING with DEVIGN_DOWNLOAD_URL.')
    if DATASET_NAME == 'bigvul' and not (WORKING_DATA_DIR / 'MSR_data_cleaned.csv').exists():
        raise FileNotFoundError('BigVul file not found. Set BIGVUL_SOURCE_PATH or enable AUTO_DOWNLOAD_DATASET_IF_MISSING with BIGVUL_DOWNLOAD_URL.')
    if DATASET_NAME == 'reveal':
        has_reveal_processed = (WORKING_DATA_DIR / 'reveal').exists()
        has_reveal_raw = (WORKING_DATA_DIR / 'reveal_raw').exists()
        if not has_reveal_processed and not has_reveal_raw:
            raise FileNotFoundError('ReVeal data was not staged. Set REVEAL_SOURCE_DIR or attach the dataset in /kaggle/input.')

    HF_HOME = WORKING_ROOT / '.cache' / 'huggingface'
    TORCH_HOME = WORKING_ROOT / '.cache' / 'torch'
    HF_HOME.mkdir(parents=True, exist_ok=True)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)

    set_or_clear_env('PYTHONUNBUFFERED', '1')
    set_or_clear_env('HF_HOME', HF_HOME)
    set_or_clear_env('TORCH_HOME', TORCH_HOME)
    set_or_clear_env('TRANSFORMERS_CACHE', HF_HOME / 'transformers')
    set_or_clear_env('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    set_or_clear_env('HUGGINGFACE_HUB_TOKEN', HF_TOKEN.strip() if HF_TOKEN else None)
    set_or_clear_env('HF_TOKEN', HF_TOKEN.strip() if HF_TOKEN else None)
    set_or_clear_env('GRACE_DATASET', DATASET_NAME)
    set_or_clear_env('GRACE_PREFILTER_MODEL_NAME', PREFILTER_MODEL_NAME)
    set_or_clear_env('GRACE_RETRIEVAL_MODEL_ID', RETRIEVAL_MODEL_ID)
    set_or_clear_env('GRACE_LOCAL_MODEL_ID', LOCAL_LLM_MODEL_ID)
    set_or_clear_env('GRACE_GRAPH_BACKEND', GRAPH_BACKEND)
    set_or_clear_env('GRACE_AUTO_DOWNLOAD_MISSING', int(bool(AUTO_DOWNLOAD_MISSING_MODELS)))
    set_or_clear_env('GRACE_AUTO_DOWNLOAD_RETRIEVAL_MODEL', int(bool(AUTO_DOWNLOAD_MISSING_MODELS)))
    set_or_clear_env('GRACE_AUTO_DOWNLOAD_MODEL', int(bool(AUTO_DOWNLOAD_MISSING_MODELS)))
    set_or_clear_env('GRACE_LOAD_IN_4BIT', int(bool(LOAD_IN_4BIT)))
    set_or_clear_env('GRACE_CALL_LLM_FOR_INSPECT', int(bool(CALL_LLM_FOR_INSPECT)))
    set_or_clear_env('GRACE_CALL_LLM_FOR_HIGH', int(bool(CALL_LLM_FOR_HIGH)))
    set_or_clear_env('GRACE_RESUME', int(bool(RESUME_RUN)))
    set_or_clear_env('GRACE_MAX_TEST_SAMPLES', MAX_TEST_SAMPLES)
    set_or_clear_env('GRACE_TEST_CHUNK_SIZE', TEST_CHUNK_SIZE)
    set_or_clear_env('GRACE_INSPECT_DEMOS', INSPECT_DEMOS)
    set_or_clear_env('GRACE_HIGH_RISK_DEMOS', HIGH_RISK_DEMOS)
    set_or_clear_env('GRACE_MAX_NEW_TOKENS', MAX_NEW_TOKENS)
    set_or_clear_env('GRACE_DEMO_CHAR_LIMIT', DEMO_CHAR_LIMIT)
    set_or_clear_env('GRACE_PROMPT_CODE_CHAR_LIMIT', PROMPT_CODE_CHAR_LIMIT)
    set_or_clear_env('GRACE_PROMPT_TOP_LINES_LIMIT', PROMPT_TOP_LINES_LIMIT)
    set_or_clear_env('GRACE_PROMPT_TOP_LINE_CHAR_LIMIT', PROMPT_TOP_LINE_CHAR_LIMIT)
    set_or_clear_env('GRACE_PROMPT_SLICES_CHAR_LIMIT', PROMPT_SLICES_CHAR_LIMIT)
    set_or_clear_env('GRACE_PROMPT_NODE_INFO_CHAR_LIMIT', PROMPT_NODE_INFO_CHAR_LIMIT)
    set_or_clear_env('GRACE_PROMPT_EDGE_INFO_CHAR_LIMIT', PROMPT_EDGE_INFO_CHAR_LIMIT)
    set_or_clear_env('GRACE_FEATURE_BATCH_SIZE', FEATURE_BATCH_SIZE)
    set_or_clear_env('GRACE_FEATURE_PROGRESS_EVERY', FEATURE_PROGRESS_EVERY)
    set_or_clear_env('GRACE_BUILD_PROGRESS_EVERY', BUILD_PROGRESS_EVERY)
    set_or_clear_env('GRACE_PREFILTER_BATCH_SIZE', PREFILTER_BATCH_SIZE)
    set_or_clear_env('GRACE_PREFILTER_EPOCHS', PREFILTER_EPOCHS)
    set_or_clear_env('GRACE_PREFILTER_LEARNING_RATE', PREFILTER_LEARNING_RATE)
    set_or_clear_env('GRACE_TARGET_RECALL', TARGET_RECALL)
    set_or_clear_env('GRACE_DIRECT_ACCEPT_MIN_PROBABILITY', DIRECT_ACCEPT_MIN_PROBABILITY)
    set_or_clear_env('GRACE_HIGH_RISK_THRESHOLD_STRATEGY', HIGH_RISK_THRESHOLD_STRATEGY)
    set_or_clear_env('GRACE_HIGH_RISK_TARGET_PRECISION', HIGH_RISK_TARGET_PRECISION)
    set_or_clear_env('GRACE_FEATURE_LIMIT', GRACE_FEATURE_LIMIT)
    set_or_clear_env('GRACE_FEATURE_LIMIT_TRAIN', GRACE_FEATURE_LIMIT_TRAIN)
    set_or_clear_env('GRACE_FEATURE_LIMIT_VAL', GRACE_FEATURE_LIMIT_VAL)
    set_or_clear_env('GRACE_FEATURE_LIMIT_TEST', GRACE_FEATURE_LIMIT_TEST)

    os.chdir(WORKING_ROOT)
    if str(BASELINE2_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINE2_DIR))

    def run_python_file(path, extra_env=None):
        path = Path(path)
        env = os.environ.copy()
        if extra_env:
            for key, value in extra_env.items():
                if value is None or value == '':
                    env.pop(key, None)
                else:
                    env[key] = str(value)
        command = [sys.executable, str(path)]
        print('Running:', ' '.join(command))
        process = subprocess.Popen(
            command,
            cwd=str(WORKING_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f'{path.name} failed with exit code {return_code}')

    def run_baseline2(script_name, extra_env=None):
        return run_python_file(BASELINE2_DIR / script_name, extra_env=extra_env)

    summary = {
        'working_root': str(WORKING_ROOT),
        'baseline2_dir': str(BASELINE2_DIR),
        'working_data_dir': str(WORKING_DATA_DIR),
        'working_shared_models_dir': str(WORKING_SHARED_MODELS_DIR),
        'dataset_download_summary': dataset_download_summary,
        'staging_summary': staging_summary,
        'previous_output_root': str(PREVIOUS_OUTPUT_ROOT) if PREVIOUS_OUTPUT_ROOT else None,
    }
    print(json.dumps(summary, indent=2))
    """


def build_runtime_sanity_cell() -> str:
    return """
    import importlib
    import json

    versions = {}
    for module_name in ['torch', 'tensorflow', 'transformers', 'bitsandbytes', 'pandas', 'numpy', 'sklearn']:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, '__version__', 'unknown')
        except Exception as exc:
            versions[module_name] = f'not available: {exc}'

    try:
        import torch
        versions['cuda_available'] = bool(torch.cuda.is_available())
        versions['cuda_device_count'] = int(torch.cuda.device_count())
        versions['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass

    report = {
        'versions': versions,
        'top_level_kaggle_inputs': [str(path) for path in sorted(KAGGLE_INPUT_ROOT.iterdir())] if KAGGLE_INPUT_ROOT.exists() else [],
    }
    print(json.dumps(report, indent=2))
    """


def build_step_run_cells() -> list[dict]:
    cells = []
    cells.append(markdown_cell("## Step 00 - Verify Assets"))
    cells.append(code_cell("run_baseline2('00_verify_assets.py')"))

    cells.append(markdown_cell("## Step 01 - Prepare Datasets"))
    cells.append(code_cell("run_baseline2('01_prepare_datasets.py')"))

    cells.append(markdown_cell("## Step 02 - Create Splits"))
    cells.append(code_cell("run_baseline2('02_create_splits.py')"))

    cells.append(markdown_cell("## Step 03 - Build Feature Store"))
    cells.append(code_cell("run_baseline2('03_build_feature_store.py')"))

    cells.append(markdown_cell("## Step 04 - Train Hybrid Prefilter"))
    cells.append(code_cell("""
    prefilter_dir = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'models' / DATASET_NAME / PREFILTER_MODEL_NAME
    prefilter_ready = (prefilter_dir / 'config.json').exists() and (prefilter_dir / 'weights.weights.h5').exists()
    if prefilter_ready:
        print(f'Skipping training because prefilter artifacts already exist at {prefilter_dir}')
    else:
        run_baseline2('04_train_hybrid_prefilter.py')
    """))

    cells.append(markdown_cell("## Step 05 - Calibrate Budget Controller"))
    cells.append(code_cell("""
    import json

    calibration_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'models' / DATASET_NAME / f'calibration.{PREFILTER_MODEL_NAME}.json'
    calibration_matches = False
    if calibration_path.exists():
        try:
            existing_calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
            calibration_matches = (
                float(existing_calibration.get('target_recall', -1.0)) == float(TARGET_RECALL)
                and float(existing_calibration.get('direct_accept_min_probability', -1.0)) == float(DIRECT_ACCEPT_MIN_PROBABILITY)
                and str(existing_calibration.get('high_risk_threshold_strategy', '')).strip().lower() == str(HIGH_RISK_THRESHOLD_STRATEGY).strip().lower()
                and float(existing_calibration.get('high_risk_target_precision', -1.0)) == float(HIGH_RISK_TARGET_PRECISION)
            )
        except Exception:
            calibration_matches = False
    if calibration_matches:
        print(f'Skipping calibration because artifact already matches config at {calibration_path}')
    else:
        run_baseline2('05_calibrate_budget_controller.py')
    """))

    cells.append(markdown_cell("## Step 06 - Build Demo Bank"))
    cells.append(code_cell("""
    demo_bank_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'retrieval' / DATASET_NAME / 'demo_bank.joblib'
    if demo_bank_path.exists():
        print(f'Skipping demo bank build because artifact already exists at {demo_bank_path}')
    else:
        run_baseline2('06_build_demo_bank.py')
    """))

    cells.append(markdown_cell("## Step 07 - Run GRACE Hybrid"))
    cells.append(code_cell("""
    import json
    import math

    from common import get_record_code, iter_jsonl

    test_path = WORKING_CODE_ROOT / 'baseline' / 'artifacts' / 'splits' / DATASET_NAME / 'test.jsonl'
    predictions_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / 'grace_hybrid_predictions.jsonl'
    run_state_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / 'grace_hybrid_run_state.json'


    def load_test_record_ids(path):
        record_ids = []
        for record in iter_jsonl(path):
            if get_record_code(record):
                record_ids.append(str(record['record_id']))
        return record_ids


    def load_prediction_ids(path):
        if not path.exists():
            return set()
        rows = []
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return {str(row['record_id']) for row in rows if row.get('schema_version') == 1}


    all_test_ids = load_test_record_ids(test_path)
    processed_ids = load_prediction_ids(predictions_path)
    remaining_ids = [record_id for record_id in all_test_ids if record_id not in processed_ids]

    if not all_test_ids:
        raise RuntimeError(f'No valid test records found at {test_path}')

    if TEST_CHUNK_SIZE is None or TEST_CHUNK_SIZE <= 0:
        print('Chunking disabled. Running Step 07 on the full unresolved test set.')
        run_baseline2(
            '07_run_grace_hybrid.py',
            extra_env={
                'GRACE_MAX_TEST_SAMPLES': None,
                'GRACE_TEST_CHUNK_SIZE': None,
                'GRACE_TEST_CHUNK_INDEX': None,
            },
        )
    else:
        num_chunks = int(math.ceil(len(all_test_ids) / TEST_CHUNK_SIZE))
        remaining_chunk_indices = []
        for chunk_index in range(num_chunks):
            chunk_ids = all_test_ids[chunk_index * TEST_CHUNK_SIZE : (chunk_index + 1) * TEST_CHUNK_SIZE]
            if any(record_id not in processed_ids for record_id in chunk_ids):
                remaining_chunk_indices.append(chunk_index)

        status = {
            'dataset': DATASET_NAME,
            'total_test_records': len(all_test_ids),
            'processed_records': len(processed_ids),
            'remaining_records': len(remaining_ids),
            'test_chunk_size': TEST_CHUNK_SIZE,
            'num_chunks': num_chunks,
            'remaining_chunk_indices': remaining_chunk_indices,
            'run_all_test_chunks_in_one_run': RUN_ALL_TEST_CHUNKS_IN_ONE_RUN,
            'predictions_path': str(predictions_path),
            'run_state_path': str(run_state_path),
        }
        print(json.dumps(status, indent=2))

        if not remaining_chunk_indices:
            print('All test chunks are already processed. Skipping Step 07.')
        else:
            chunk_indices_to_run = remaining_chunk_indices if RUN_ALL_TEST_CHUNKS_IN_ONE_RUN else [remaining_chunk_indices[0]]
            for chunk_index in chunk_indices_to_run:
                print(f'Running chunk {chunk_index + 1}/{num_chunks} with chunk_size={TEST_CHUNK_SIZE}')
                run_baseline2(
                    '07_run_grace_hybrid.py',
                    extra_env={
                        'GRACE_MAX_TEST_SAMPLES': None,
                        'GRACE_TEST_CHUNK_SIZE': TEST_CHUNK_SIZE,
                        'GRACE_TEST_CHUNK_INDEX': chunk_index,
                    },
                )
    """))

    cells.append(markdown_cell("## Step 08 - Evaluate Predictions"))
    cells.append(code_cell("""
    import json

    run_state_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / 'grace_hybrid_run_state.json'
    if not run_state_path.exists():
        print(f'Skipping evaluation because run_state does not exist yet: {run_state_path}')
    else:
        run_state = json.loads(run_state_path.read_text(encoding='utf-8'))
        if bool(run_state.get('complete')):
            run_baseline2('08_evaluate_predictions.py')
        else:
            payload = {
                'message': 'Skipping evaluation because not all test chunks are complete yet.',
                'resolved_samples': run_state.get('resolved_samples'),
                'target_samples': run_state.get('target_samples'),
                'chunking': run_state.get('chunking'),
                'predictions_path': run_state.get('predictions_path'),
            }
            print(json.dumps(payload, indent=2))
    """))

    cells.append(markdown_cell("## Final Summary"))
    cells.append(code_cell("""
    import json
    import math

    metrics_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'metrics' / DATASET_NAME / 'grace_hybrid_evaluation_summary.json'
    run_state_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / 'grace_hybrid_run_state.json'
    predictions_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / 'grace_hybrid_predictions.jsonl'

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        preview = {
            'dataset': metrics.get('dataset'),
            'samples': metrics.get('samples'),
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1': metrics.get('f1'),
            'roc_auc': metrics.get('roc_auc'),
            'pr_auc': metrics.get('pr_auc'),
            'llm_calls': metrics.get('llm_calls'),
            'llm_call_ratio': metrics.get('llm_call_ratio'),
            'routing': metrics.get('routing'),
            'decision_sources': metrics.get('decision_sources'),
            'metrics_path': str(metrics_path),
            'run_state_path': str(run_state_path),
            'predictions_path': str(predictions_path),
        }
        print(json.dumps(preview, indent=2))
    elif run_state_path.exists():
        run_state = json.loads(run_state_path.read_text(encoding='utf-8'))
        target_samples = int(run_state.get('target_samples') or 0)
        resolved_samples = int(run_state.get('resolved_samples') or 0)
        remaining = max(0, target_samples - resolved_samples)
        chunk_size = int((run_state.get('chunking') or {}).get('chunk_size') or TEST_CHUNK_SIZE or 0)
        remaining_chunks_estimate = int(math.ceil(remaining / chunk_size)) if chunk_size > 0 else None
        preview = {
            'message': 'Evaluation summary is not available yet because full test-set chunking is still in progress.',
            'dataset': run_state.get('dataset'),
            'resolved_samples': resolved_samples,
            'target_samples': target_samples,
            'remaining_samples': remaining,
            'remaining_chunks_estimate': remaining_chunks_estimate,
            'chunking': run_state.get('chunking'),
            'predictions_path': str(predictions_path),
            'run_state_path': str(run_state_path),
        }
        print(json.dumps(preview, indent=2))
    else:
        raise FileNotFoundError(f'Neither metrics nor run_state file exists yet under {predictions_path.parent}')
    """))
    return cells


def build_notebook() -> dict:
    cells = [
        markdown_cell("""
        # FINAL Kaggle Notebook for GRACE Baseline 2

        This notebook is self-contained. All `baseline2` module and step sources are written directly into the notebook and then materialized into files under `/kaggle/working` using `%%writefile`.

        What it does:
        - configures the run in one place,
        - installs missing packages,
        - prepares the Kaggle workspace,
        - writes all `baseline2` code from notebook cells into the working directory,
        - auto-downloads `devign` when needed,
        - restores prior artifacts from a previous Kaggle output dataset when available,
        - runs steps `00` to `08`,
        - prints the final evaluation summary.

        The notebook is preconfigured for `devign` and chunked full-test execution on Kaggle.
        It will automatically run all remaining chunks in sequence inside one session. If Kaggle stops the session early, save the output as a Kaggle dataset, attach that output dataset on the next run, and the notebook will auto-resume from the next unfinished chunk.
        """),
        code_cell(build_config_cell()),
        markdown_cell("""
        ## Install Missing Packages

        This installs only missing lightweight packages and leaves heavyweight packages such as `torch` and `tensorflow` to the Kaggle image.
        """),
        code_cell(build_install_cell()),
        markdown_cell("""
        ## Prepare Kaggle Workspace

        This prepares the Kaggle workspace, stages datasets and model directories, restores prior artifacts, sets environment variables, and defines helpers to run the generated Python files.
        """),
        code_cell(build_setup_cell()),
        markdown_cell("""
        ## Materialize Baseline2 Source Files

        The next cells write the exact `baseline2` source files into `/kaggle/working` before the pipeline runs.
        """),
    ]

    for filename in MODULE_FILES:
        cells.append(markdown_cell(f"### File `{filename}`"))
        cells.append(writefile_cell(f"GRACE-improve/baseline/baseline2/{filename}", read_text(BASELINE2_DIR / filename)))

    for filename in STEP_FILES:
        cells.append(markdown_cell(f"### File `{filename}`"))
        cells.append(writefile_cell(f"GRACE-improve/baseline/baseline2/{filename}", read_text(BASELINE2_DIR / filename)))

    cells.append(markdown_cell("## Runtime Sanity Check"))
    cells.append(code_cell(build_runtime_sanity_cell()))
    cells.extend(build_step_run_cells())

    return {
        "cells": cells,
        "metadata": load_notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    FINAL_NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {FINAL_NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
