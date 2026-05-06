import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_NOTEBOOK = ROOT / "FINAL.ipynb"

WRITEFILE_SOURCES = [
    "common.py",
    "datasets.py",
    "graphs.py",
    "metrics.py",
    "localizer.py",
    "retrieval.py",
    "hybrid_prefilter.py",
    "local_llm_client.py",
    "evaluate_predictions.py",
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


VARIANTS = [
    {
        "name": "B1",
        "title": "Calibration only",
        "notes": "Ablation focused on calibration selection while keeping the baseline routing structure.",
        "model_suffix": "b1",
        "target_recall": 0.995,
        "direct_accept_min_probability": 0.20,
        "high_risk_threshold_strategy": "f1",
        "high_risk_target_precision": 0.70,
        "calibration_method": "auto",
        "routing_mode": "baseline",
        "routing_objective": "f1",
        "routing_inspect_proxy": "probability",
        "routing_recall_floor": 0.995,
        "llm_budget": 0.1592,
        "tau_neg_min": 0.02,
        "tau_neg_max": 0.30,
        "tau_neg_steps": 15,
        "tau_pos_min": 0.45,
        "tau_pos_max": 0.90,
        "tau_pos_steps": 19,
        "force_rebuild_features": False,
        "feature_store_suffix": "",
        "hard_negative_mining": False,
        "hard_negative_quantile": 0.85,
        "hard_negative_weight": 2.5,
        "hard_negative_epochs": 2,
        "prefilter_loss": "bce",
        "prefilter_focal_gamma": 2.0,
        "token_max_tokens": 32000,
        "token_sequence_length": 384,
        "token_embedding_dim": 96,
        "token_filters": 96,
        "ast_max_tokens": 8000,
        "ast_sequence_length": 196,
        "ast_embedding_dim": 64,
        "ast_filters": 64,
        "projection_dim": 192,
        "dense_units": 192,
        "dropout_rate": 0.25,
        "prefilter_epochs": 10,
        "prefilter_learning_rate": 7e-4,
        "evidence_aware_verifier": False,
        "max_new_tokens": 96,
        "prompt_top_lines_limit": 3,
        "prompt_top_line_char_limit": 120,
        "prompt_slices_char_limit": 900,
        "prompt_node_info_char_limit": 900,
        "prompt_edge_info_char_limit": 900,
        "call_llm_for_high": False,
        "graph_backend": "stable",
    },
    {
        "name": "B2",
        "title": "Calibration plus constrained routing",
        "notes": "Adds threshold search with recall and budget constraints over the calibrated scores.",
        "model_suffix": "b2",
        "target_recall": 0.915,
        "direct_accept_min_probability": 0.20,
        "high_risk_threshold_strategy": "f1",
        "high_risk_target_precision": 0.70,
        "calibration_method": "auto",
        "routing_mode": "constrained",
        "routing_objective": "accuracy",
        "routing_inspect_proxy": "probability",
        "routing_recall_floor": 0.915,
        "llm_budget": 0.15,
        "tau_neg_min": 0.02,
        "tau_neg_max": 0.30,
        "tau_neg_steps": 15,
        "tau_pos_min": 0.45,
        "tau_pos_max": 0.90,
        "tau_pos_steps": 19,
        "force_rebuild_features": False,
        "feature_store_suffix": "",
        "hard_negative_mining": False,
        "hard_negative_quantile": 0.85,
        "hard_negative_weight": 2.5,
        "hard_negative_epochs": 2,
        "prefilter_loss": "bce",
        "prefilter_focal_gamma": 2.0,
        "token_max_tokens": 32000,
        "token_sequence_length": 384,
        "token_embedding_dim": 96,
        "token_filters": 96,
        "ast_max_tokens": 8000,
        "ast_sequence_length": 196,
        "ast_embedding_dim": 64,
        "ast_filters": 64,
        "projection_dim": 192,
        "dense_units": 192,
        "dropout_rate": 0.25,
        "prefilter_epochs": 10,
        "prefilter_learning_rate": 7e-4,
        "evidence_aware_verifier": False,
        "max_new_tokens": 96,
        "prompt_top_lines_limit": 3,
        "prompt_top_line_char_limit": 120,
        "prompt_slices_char_limit": 900,
        "prompt_node_info_char_limit": 900,
        "prompt_edge_info_char_limit": 900,
        "call_llm_for_high": False,
        "graph_backend": "stable",
    },
    {
        "name": "B3",
        "title": "Routing plus hard-negative mining",
        "notes": "Keeps constrained routing and adds a second pass that upweights hard negatives during prefilter training.",
        "model_suffix": "b3",
        "target_recall": 0.915,
        "direct_accept_min_probability": 0.20,
        "high_risk_threshold_strategy": "f1",
        "high_risk_target_precision": 0.70,
        "calibration_method": "auto",
        "routing_mode": "constrained",
        "routing_objective": "accuracy",
        "routing_inspect_proxy": "probability",
        "routing_recall_floor": 0.915,
        "llm_budget": 0.15,
        "tau_neg_min": 0.02,
        "tau_neg_max": 0.30,
        "tau_neg_steps": 15,
        "tau_pos_min": 0.45,
        "tau_pos_max": 0.90,
        "tau_pos_steps": 19,
        "force_rebuild_features": False,
        "feature_store_suffix": "",
        "hard_negative_mining": True,
        "hard_negative_quantile": 0.85,
        "hard_negative_weight": 2.5,
        "hard_negative_epochs": 2,
        "prefilter_loss": "bce",
        "prefilter_focal_gamma": 2.0,
        "token_max_tokens": 32000,
        "token_sequence_length": 384,
        "token_embedding_dim": 96,
        "token_filters": 96,
        "ast_max_tokens": 8000,
        "ast_sequence_length": 196,
        "ast_embedding_dim": 64,
        "ast_filters": 64,
        "projection_dim": 192,
        "dense_units": 192,
        "dropout_rate": 0.25,
        "prefilter_epochs": 10,
        "prefilter_learning_rate": 7e-4,
        "evidence_aware_verifier": False,
        "max_new_tokens": 96,
        "prompt_top_lines_limit": 3,
        "prompt_top_line_char_limit": 120,
        "prompt_slices_char_limit": 900,
        "prompt_node_info_char_limit": 900,
        "prompt_edge_info_char_limit": 900,
        "call_llm_for_high": False,
        "graph_backend": "stable",
    },
    {
        "name": "B4",
        "title": "Hard negatives plus stable graph backend",
        "notes": "Forces a stable graph extraction path and rebuilds feature stores under a separate suffix.",
        "model_suffix": "b4",
        "target_recall": 0.915,
        "direct_accept_min_probability": 0.20,
        "high_risk_threshold_strategy": "f1",
        "high_risk_target_precision": 0.70,
        "calibration_method": "auto",
        "routing_mode": "constrained",
        "routing_objective": "accuracy",
        "routing_inspect_proxy": "probability",
        "routing_recall_floor": 0.915,
        "llm_budget": 0.15,
        "tau_neg_min": 0.02,
        "tau_neg_max": 0.30,
        "tau_neg_steps": 15,
        "tau_pos_min": 0.45,
        "tau_pos_max": 0.90,
        "tau_pos_steps": 19,
        "force_rebuild_features": True,
        "feature_store_suffix": "stable_graph",
        "hard_negative_mining": True,
        "hard_negative_quantile": 0.80,
        "hard_negative_weight": 3.0,
        "hard_negative_epochs": 2,
        "prefilter_loss": "bce",
        "prefilter_focal_gamma": 2.0,
        "token_max_tokens": 32000,
        "token_sequence_length": 384,
        "token_embedding_dim": 96,
        "token_filters": 96,
        "ast_max_tokens": 8000,
        "ast_sequence_length": 196,
        "ast_embedding_dim": 64,
        "ast_filters": 64,
        "projection_dim": 192,
        "dense_units": 192,
        "dropout_rate": 0.25,
        "prefilter_epochs": 10,
        "prefilter_learning_rate": 7e-4,
        "evidence_aware_verifier": False,
        "max_new_tokens": 96,
        "prompt_top_lines_limit": 3,
        "prompt_top_line_char_limit": 120,
        "prompt_slices_char_limit": 900,
        "prompt_node_info_char_limit": 900,
        "prompt_edge_info_char_limit": 900,
        "call_llm_for_high": False,
        "graph_backend": "stable",
    },
    {
        "name": "B5",
        "title": "Stable graph backend plus prefilter v2",
        "notes": "Upgrades the prefilter capacity and switches the training loss to focal loss.",
        "model_suffix": "b5",
        "target_recall": 0.915,
        "direct_accept_min_probability": 0.20,
        "high_risk_threshold_strategy": "f1",
        "high_risk_target_precision": 0.70,
        "calibration_method": "auto",
        "routing_mode": "constrained",
        "routing_objective": "accuracy",
        "routing_inspect_proxy": "probability",
        "routing_recall_floor": 0.915,
        "llm_budget": 0.15,
        "tau_neg_min": 0.02,
        "tau_neg_max": 0.30,
        "tau_neg_steps": 15,
        "tau_pos_min": 0.45,
        "tau_pos_max": 0.90,
        "tau_pos_steps": 19,
        "force_rebuild_features": True,
        "feature_store_suffix": "stable_graph",
        "hard_negative_mining": True,
        "hard_negative_quantile": 0.80,
        "hard_negative_weight": 3.0,
        "hard_negative_epochs": 2,
        "prefilter_loss": "focal",
        "prefilter_focal_gamma": 2.0,
        "token_max_tokens": 32000,
        "token_sequence_length": 512,
        "token_embedding_dim": 128,
        "token_filters": 128,
        "ast_max_tokens": 8000,
        "ast_sequence_length": 256,
        "ast_embedding_dim": 96,
        "ast_filters": 96,
        "projection_dim": 256,
        "dense_units": 256,
        "dropout_rate": 0.30,
        "prefilter_epochs": 12,
        "prefilter_learning_rate": 7e-4,
        "evidence_aware_verifier": False,
        "max_new_tokens": 96,
        "prompt_top_lines_limit": 3,
        "prompt_top_line_char_limit": 120,
        "prompt_slices_char_limit": 900,
        "prompt_node_info_char_limit": 900,
        "prompt_edge_info_char_limit": 900,
        "call_llm_for_high": False,
        "graph_backend": "stable",
    },
    {
        "name": "B6",
        "title": "Prefilter v2 plus evidence-aware verifier",
        "notes": "Adds a stricter verifier schema that requires concrete code evidence before accepting a positive LLM answer.",
        "model_suffix": "b6",
        "target_recall": 0.915,
        "direct_accept_min_probability": 0.20,
        "high_risk_threshold_strategy": "f1",
        "high_risk_target_precision": 0.70,
        "calibration_method": "auto",
        "routing_mode": "constrained",
        "routing_objective": "accuracy",
        "routing_inspect_proxy": "probability",
        "routing_recall_floor": 0.915,
        "llm_budget": 0.15,
        "tau_neg_min": 0.02,
        "tau_neg_max": 0.30,
        "tau_neg_steps": 15,
        "tau_pos_min": 0.45,
        "tau_pos_max": 0.90,
        "tau_pos_steps": 19,
        "force_rebuild_features": True,
        "feature_store_suffix": "stable_graph",
        "hard_negative_mining": True,
        "hard_negative_quantile": 0.80,
        "hard_negative_weight": 3.0,
        "hard_negative_epochs": 2,
        "prefilter_loss": "focal",
        "prefilter_focal_gamma": 2.0,
        "token_max_tokens": 32000,
        "token_sequence_length": 512,
        "token_embedding_dim": 128,
        "token_filters": 128,
        "ast_max_tokens": 8000,
        "ast_sequence_length": 256,
        "ast_embedding_dim": 96,
        "ast_filters": 96,
        "projection_dim": 256,
        "dense_units": 256,
        "dropout_rate": 0.30,
        "prefilter_epochs": 12,
        "prefilter_learning_rate": 7e-4,
        "evidence_aware_verifier": True,
        "max_new_tokens": 160,
        "prompt_top_lines_limit": 5,
        "prompt_top_line_char_limit": 140,
        "prompt_slices_char_limit": 1100,
        "prompt_node_info_char_limit": 1100,
        "prompt_edge_info_char_limit": 1100,
        "call_llm_for_high": False,
        "graph_backend": "stable",
    },
]


def load_notebook() -> dict:
    return json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))


def to_source(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_cell_text(cell: dict, text: str) -> None:
    cell["source"] = to_source(text)


def find_code_cell(nb: dict, startswith: str) -> dict:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and cell_text(cell).startswith(startswith):
            return cell
    raise KeyError(f"Could not find code cell starting with {startswith!r}")


def clear_outputs(nb: dict) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def replace_writefile_cell(nb: dict, file_name: str, source_text: str) -> None:
    prefix = f"%%writefile GRACE-improve/baseline/baseline2/{file_name}"
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and cell_text(cell).startswith(prefix):
            set_cell_text(cell, f"{prefix}\n{source_text.rstrip()}\n")
            return
    raise KeyError(f"Could not find writefile cell for {file_name}")


def insert_variant_cells(nb: dict, variant: dict) -> None:
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(
            f"## Variant {variant['name']}\n\n"
            f"{variant['title']}\n\n"
            f"{variant['notes']}"
        ),
    }
    code = f"""# Variant override injected after the base Kaggle config.
EXPERIMENT_VARIANT = {variant['name']!r}
EXPERIMENT_NAME = {variant['title']!r}
EXPERIMENT_NOTES = {variant['notes']!r}

BASE_PREFILTER_MODEL_NAME = PREFILTER_MODEL_NAME
VARIANT_OUTPUT_SUFFIX = EXPERIMENT_VARIANT.lower()
PREFILTER_MODEL_NAME = f"{{BASE_PREFILTER_MODEL_NAME}}_{{VARIANT_OUTPUT_SUFFIX}}"

GRACE_EXPERIMENT_VARIANT = EXPERIMENT_VARIANT
GRACE_VARIANT_OUTPUT_SUFFIX = VARIANT_OUTPUT_SUFFIX
GRACE_PREDICTION_FILE_STEM = f"grace_hybrid_predictions_{{VARIANT_OUTPUT_SUFFIX}}"
GRACE_RUN_STATE_FILE_STEM = f"grace_hybrid_run_state_{{VARIANT_OUTPUT_SUFFIX}}"
GRACE_EVALUATION_FILE_STEM = f"grace_hybrid_evaluation_summary_{{VARIANT_OUTPUT_SUFFIX}}"

TARGET_RECALL = {variant['target_recall']!r}
DIRECT_ACCEPT_MIN_PROBABILITY = {variant['direct_accept_min_probability']!r}
HIGH_RISK_THRESHOLD_STRATEGY = {variant['high_risk_threshold_strategy']!r}
HIGH_RISK_TARGET_PRECISION = {variant['high_risk_target_precision']!r}
GRAPH_BACKEND = {variant['graph_backend']!r}

GRACE_CALIBRATION_METHOD = {variant['calibration_method']!r}
GRACE_ROUTING_MODE = {variant['routing_mode']!r}
GRACE_ROUTING_OBJECTIVE = {variant['routing_objective']!r}
GRACE_ROUTING_INSPECT_PROXY = {variant['routing_inspect_proxy']!r}
GRACE_ROUTING_RECALL_FLOOR = {variant['routing_recall_floor']!r}
GRACE_LLM_BUDGET = {variant['llm_budget']!r}
GRACE_TAU_NEG_MIN = {variant['tau_neg_min']!r}
GRACE_TAU_NEG_MAX = {variant['tau_neg_max']!r}
GRACE_TAU_NEG_STEPS = {variant['tau_neg_steps']!r}
GRACE_TAU_POS_MIN = {variant['tau_pos_min']!r}
GRACE_TAU_POS_MAX = {variant['tau_pos_max']!r}
GRACE_TAU_POS_STEPS = {variant['tau_pos_steps']!r}
GRACE_FORCE_REBUILD_FEATURES = {variant['force_rebuild_features']!r}
GRACE_FEATURE_STORE_SUFFIX = {variant['feature_store_suffix']!r}
GRACE_HARD_NEGATIVE_MINING = {variant['hard_negative_mining']!r}
GRACE_HARD_NEGATIVE_QUANTILE = {variant['hard_negative_quantile']!r}
GRACE_HARD_NEGATIVE_WEIGHT = {variant['hard_negative_weight']!r}
GRACE_HARD_NEGATIVE_EPOCHS = {variant['hard_negative_epochs']!r}
GRACE_PREFILTER_LOSS = {variant['prefilter_loss']!r}
GRACE_PREFILTER_FOCAL_GAMMA = {variant['prefilter_focal_gamma']!r}
GRACE_TOKEN_MAX_TOKENS = {variant['token_max_tokens']!r}
GRACE_TOKEN_SEQUENCE_LENGTH = {variant['token_sequence_length']!r}
GRACE_TOKEN_EMBEDDING_DIM = {variant['token_embedding_dim']!r}
GRACE_TOKEN_FILTERS = {variant['token_filters']!r}
GRACE_AST_MAX_TOKENS = {variant['ast_max_tokens']!r}
GRACE_AST_SEQUENCE_LENGTH = {variant['ast_sequence_length']!r}
GRACE_AST_EMBEDDING_DIM = {variant['ast_embedding_dim']!r}
GRACE_AST_FILTERS = {variant['ast_filters']!r}
GRACE_PREFILTER_PROJECTION_DIM = {variant['projection_dim']!r}
GRACE_PREFILTER_DENSE_UNITS = {variant['dense_units']!r}
GRACE_PREFILTER_DROPOUT = {variant['dropout_rate']!r}
GRACE_PREFILTER_EPOCHS = {variant['prefilter_epochs']!r}
GRACE_PREFILTER_LEARNING_RATE = {variant['prefilter_learning_rate']!r}
GRACE_EVIDENCE_AWARE_VERIFIER = {variant['evidence_aware_verifier']!r}
GRACE_MAX_NEW_TOKENS = {variant['max_new_tokens']!r}
GRACE_PROMPT_TOP_LINES_LIMIT = {variant['prompt_top_lines_limit']!r}
GRACE_PROMPT_TOP_LINE_CHAR_LIMIT = {variant['prompt_top_line_char_limit']!r}
GRACE_PROMPT_SLICES_CHAR_LIMIT = {variant['prompt_slices_char_limit']!r}
GRACE_PROMPT_NODE_INFO_CHAR_LIMIT = {variant['prompt_node_info_char_limit']!r}
GRACE_PROMPT_EDGE_INFO_CHAR_LIMIT = {variant['prompt_edge_info_char_limit']!r}
GRACE_CALL_LLM_FOR_HIGH = {variant['call_llm_for_high']!r}
"""
    code_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }
    nb["cells"][2:2] = [md_cell, code_cell]


def patch_setup_cell(nb: dict) -> None:
    cell = find_code_cell(nb, "import gzip")
    text = cell_text(cell)
    anchor = "set_or_clear_env('GRACE_FEATURE_LIMIT_TEST', GRACE_FEATURE_LIMIT_TEST)\n"
    if anchor not in text:
        raise RuntimeError("Could not find environment export anchor in setup cell.")
    extra = """set_or_clear_env('GRACE_EXPERIMENT_VARIANT', globals().get('GRACE_EXPERIMENT_VARIANT'))
set_or_clear_env('GRACE_VARIANT_OUTPUT_SUFFIX', globals().get('GRACE_VARIANT_OUTPUT_SUFFIX'))
set_or_clear_env('GRACE_PREDICTION_FILE_STEM', globals().get('GRACE_PREDICTION_FILE_STEM'))
set_or_clear_env('GRACE_RUN_STATE_FILE_STEM', globals().get('GRACE_RUN_STATE_FILE_STEM'))
set_or_clear_env('GRACE_EVALUATION_FILE_STEM', globals().get('GRACE_EVALUATION_FILE_STEM'))
set_or_clear_env('GRACE_CALIBRATION_METHOD', globals().get('GRACE_CALIBRATION_METHOD'))
set_or_clear_env('GRACE_ROUTING_MODE', globals().get('GRACE_ROUTING_MODE'))
set_or_clear_env('GRACE_ROUTING_OBJECTIVE', globals().get('GRACE_ROUTING_OBJECTIVE'))
set_or_clear_env('GRACE_ROUTING_INSPECT_PROXY', globals().get('GRACE_ROUTING_INSPECT_PROXY'))
set_or_clear_env('GRACE_ROUTING_RECALL_FLOOR', globals().get('GRACE_ROUTING_RECALL_FLOOR'))
set_or_clear_env('GRACE_LLM_BUDGET', globals().get('GRACE_LLM_BUDGET'))
set_or_clear_env('GRACE_TAU_NEG_MIN', globals().get('GRACE_TAU_NEG_MIN'))
set_or_clear_env('GRACE_TAU_NEG_MAX', globals().get('GRACE_TAU_NEG_MAX'))
set_or_clear_env('GRACE_TAU_NEG_STEPS', globals().get('GRACE_TAU_NEG_STEPS'))
set_or_clear_env('GRACE_TAU_POS_MIN', globals().get('GRACE_TAU_POS_MIN'))
set_or_clear_env('GRACE_TAU_POS_MAX', globals().get('GRACE_TAU_POS_MAX'))
set_or_clear_env('GRACE_TAU_POS_STEPS', globals().get('GRACE_TAU_POS_STEPS'))
set_or_clear_env('GRACE_FORCE_REBUILD_FEATURES', globals().get('GRACE_FORCE_REBUILD_FEATURES'))
set_or_clear_env('GRACE_FEATURE_STORE_SUFFIX', globals().get('GRACE_FEATURE_STORE_SUFFIX'))
set_or_clear_env('GRACE_HARD_NEGATIVE_MINING', globals().get('GRACE_HARD_NEGATIVE_MINING'))
set_or_clear_env('GRACE_HARD_NEGATIVE_QUANTILE', globals().get('GRACE_HARD_NEGATIVE_QUANTILE'))
set_or_clear_env('GRACE_HARD_NEGATIVE_WEIGHT', globals().get('GRACE_HARD_NEGATIVE_WEIGHT'))
set_or_clear_env('GRACE_HARD_NEGATIVE_EPOCHS', globals().get('GRACE_HARD_NEGATIVE_EPOCHS'))
set_or_clear_env('GRACE_PREFILTER_LOSS', globals().get('GRACE_PREFILTER_LOSS'))
set_or_clear_env('GRACE_PREFILTER_FOCAL_GAMMA', globals().get('GRACE_PREFILTER_FOCAL_GAMMA'))
set_or_clear_env('GRACE_TOKEN_MAX_TOKENS', globals().get('GRACE_TOKEN_MAX_TOKENS'))
set_or_clear_env('GRACE_TOKEN_SEQUENCE_LENGTH', globals().get('GRACE_TOKEN_SEQUENCE_LENGTH'))
set_or_clear_env('GRACE_TOKEN_EMBEDDING_DIM', globals().get('GRACE_TOKEN_EMBEDDING_DIM'))
set_or_clear_env('GRACE_TOKEN_FILTERS', globals().get('GRACE_TOKEN_FILTERS'))
set_or_clear_env('GRACE_AST_MAX_TOKENS', globals().get('GRACE_AST_MAX_TOKENS'))
set_or_clear_env('GRACE_AST_SEQUENCE_LENGTH', globals().get('GRACE_AST_SEQUENCE_LENGTH'))
set_or_clear_env('GRACE_AST_EMBEDDING_DIM', globals().get('GRACE_AST_EMBEDDING_DIM'))
set_or_clear_env('GRACE_AST_FILTERS', globals().get('GRACE_AST_FILTERS'))
set_or_clear_env('GRACE_PREFILTER_PROJECTION_DIM', globals().get('GRACE_PREFILTER_PROJECTION_DIM'))
set_or_clear_env('GRACE_PREFILTER_DENSE_UNITS', globals().get('GRACE_PREFILTER_DENSE_UNITS'))
set_or_clear_env('GRACE_PREFILTER_DROPOUT', globals().get('GRACE_PREFILTER_DROPOUT'))
set_or_clear_env('GRACE_PREFILTER_EPOCHS', globals().get('GRACE_PREFILTER_EPOCHS'))
set_or_clear_env('GRACE_PREFILTER_LEARNING_RATE', globals().get('GRACE_PREFILTER_LEARNING_RATE'))
set_or_clear_env('GRACE_EVIDENCE_AWARE_VERIFIER', globals().get('GRACE_EVIDENCE_AWARE_VERIFIER'))
"""
    set_cell_text(cell, text.replace(anchor, anchor + extra))


def patch_variant_sensitive_cells(nb: dict) -> None:
    cell_60 = find_code_cell(nb, "import json\nimport math\n\nfrom common import get_record_code, iter_jsonl\n")
    set_cell_text(
        cell_60,
        """import json
import math

from common import get_record_code, iter_jsonl

variant_suffix = globals().get('GRACE_VARIANT_OUTPUT_SUFFIX', '')
prediction_file_stem = globals().get('GRACE_PREDICTION_FILE_STEM') or (f'grace_hybrid_predictions_{variant_suffix}' if variant_suffix else 'grace_hybrid_predictions')
run_state_file_stem = globals().get('GRACE_RUN_STATE_FILE_STEM') or (f'grace_hybrid_run_state_{variant_suffix}' if variant_suffix else 'grace_hybrid_run_state')
test_path = WORKING_CODE_ROOT / 'baseline' / 'artifacts' / 'splits' / DATASET_NAME / 'test.jsonl'
predictions_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / f'{prediction_file_stem}.jsonl'
run_state_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / f'{run_state_file_stem}.json'


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
""",
    )

    cell_62 = find_code_cell(nb, "import json\n\nrun_state_path = WORKING_CODE_ROOT")
    set_cell_text(
        cell_62,
        """import json

variant_suffix = globals().get('GRACE_VARIANT_OUTPUT_SUFFIX', '')
prediction_file_stem = globals().get('GRACE_PREDICTION_FILE_STEM') or (f'grace_hybrid_predictions_{variant_suffix}' if variant_suffix else 'grace_hybrid_predictions')
run_state_file_stem = globals().get('GRACE_RUN_STATE_FILE_STEM') or (f'grace_hybrid_run_state_{variant_suffix}' if variant_suffix else 'grace_hybrid_run_state')
run_state_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / f'{run_state_file_stem}.json'
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
""",
    )

    cell_64 = find_code_cell(nb, "import json\nimport math\n\nmetrics_path = WORKING_CODE_ROOT")
    set_cell_text(
        cell_64,
        """import json
import math

variant_suffix = globals().get('GRACE_VARIANT_OUTPUT_SUFFIX', '')
prediction_file_stem = globals().get('GRACE_PREDICTION_FILE_STEM') or (f'grace_hybrid_predictions_{variant_suffix}' if variant_suffix else 'grace_hybrid_predictions')
run_state_file_stem = globals().get('GRACE_RUN_STATE_FILE_STEM') or (f'grace_hybrid_run_state_{variant_suffix}' if variant_suffix else 'grace_hybrid_run_state')
metrics_file_stem = globals().get('GRACE_EVALUATION_FILE_STEM') or (f'grace_hybrid_evaluation_summary_{variant_suffix}' if variant_suffix else 'grace_hybrid_evaluation_summary')
metrics_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'metrics' / DATASET_NAME / f'{metrics_file_stem}.json'
run_state_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / f'{run_state_file_stem}.json'
predictions_path = WORKING_CODE_ROOT / 'baseline' / 'baseline2' / 'artifacts' / 'predictions' / DATASET_NAME / f'{prediction_file_stem}.jsonl'

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
""",
    )


def rewrite_writefiles(nb: dict) -> None:
    for file_name in WRITEFILE_SOURCES:
        replace_writefile_cell(nb, file_name, (ROOT / file_name).read_text(encoding="utf-8").replace("\r\n", "\n"))


def build_variant_notebook(variant: dict) -> dict:
    nb = copy.deepcopy(load_notebook())
    rewrite_writefiles(nb)
    insert_variant_cells(nb, variant)
    patch_setup_cell(nb)
    patch_variant_sensitive_cells(nb)
    clear_outputs(nb)
    return nb


def main() -> None:
    for variant in VARIANTS:
        nb = build_variant_notebook(variant)
        out_path = ROOT / f"FINAL_{variant['name'][1:]}.ipynb"
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
