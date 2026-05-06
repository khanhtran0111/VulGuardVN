# Baseline cho GRACE-improve

Baseline nay giu y tuong chen mot tang deep learning pre-filter o dau pipeline, nhung da duoc dua tro lai gan voi kien truc GRACE goc:

`prefilter -> CodeT5 retrieval -> Joern graph prompt -> local VulnLLM-R-7B`

## Pipeline

Pipeline hien tai gom 4 tang chinh:

1. `Pre-filter`

- Dung `Token-CNN` train tren train split.
- Dau ra la `prefilter_probability`, sau do duoc calibration bang `Platt scaling`.
- Router chia mau thanh 3 nhom:
  - `skip`: xac suat thap, ket luan `Non-vulnerable` ma khong goi LLM.
  - `uncertain`: goi retrieval + graph + LLM.
  - `high`: goi retrieval + graph + LLM voi nhieu demonstration hon.

2. `Demonstration Retrieval`

- Demo bank duoc build tu train split bang `05_build_demo_bank.py`.
- Semantic shortlist uu tien `CodeT5 encoder`.
- Sau shortlist, he thong rerank bang:
  - lexical similarity: `Jaccard` tren token code
  - syntactic similarity: `Levenshtein` tren `AST sequence`
- Neu runtime `CodeT5` khong san sang, backend `auto` se fallback ve `TF-IDF`.

3. `Graph Structure`

- Graph module uu tien `Joern`.
- Hien tai he thong extract va prompt 3 nguon cau truc gan paper:
  - `AST`
  - `CFG`
  - `PDG`
- Dau ra duoc chuan hoa thanh:
  - `node_info`
  - `edge_info`
  - `ast_sequence`
- Neu `Joern` khong co, module se fallback ve graph heuristic va ghi ro backend trong artifact/prediction summary.

4. `Enhanced Detection`

- Prompt dua vao local LLM:
  - code function
  - node information
  - edge information
  - retrieved demonstrations
  - prefilter risk band
- Local LLM mac dinh la `Virtue-AI-HUB/VulnLLM-R-7B`.
- Dau ra cuoi cung van duoc chuan hoa ve JSON 1 dong de de parse va evaluate.

## Cau truc file

- `00_download_reveal.py`: tai va giai nen ReVeal tu nguon chinh thuc.
- `00_5_prepare_reveal_parquet.py`: convert ReVeal parquet thanh `jsonl`.
- `01_prepare_datasets.py`: index metadata cua Devign, Big-Vul va ReVeal.
- `02_create_splits.py`: tao split `train/val/test`; ReVeal giu official split neu co.
- `03_train_prefilter.ipynb`: train `Token-CNN` pre-filter.
- `04_calibrate_prefilter.py`: fit `Platt scaling` va tim `tau_low`, `tau_high`.
- `05_build_demo_bank.py`: build retrieval bank tu train split.
- `06_run_grace_prefilter.py`: chay full cascade `prefilter -> retrieval -> graph -> local LLM`.
- `07_evaluate_predictions.py`: tong hop metrics va timing breakdown.
- `retrieval.py`: semantic retrieval bang `CodeT5` hoac fallback `TF-IDF`.
- `graphs.py`: extract/caching graph structure, uu tien `Joern`.
- `local_llm_client.py`: prompt handler va local inference cho `VulnLLM-R-7B`.
- `download_codet5_retrieval_model.py`: tai checkpoint `CodeT5`.
- `download_joern.py`: tai va giai nen `Joern` vao artifact local.
- `download_vulnllm_r_7b.py`: tai local LLM.

## Yeu cau

- Python packages:
  - `tensorflow`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `python-dotenv`
  - `joblib`
  - `huggingface_hub`
  - `torch`
  - `transformers`
  - `accelerate`
  - `bitsandbytes`
  - `sentencepiece`

- Java:
  - `Joern` can Java runtime. Java 21 dang hoat dong tot voi setup hien tai.

- File `.env` tai `GRACE-improve/.env`:

```env
HF_TOKEN="optional_if_you_need_it"
```

`HF_TOKEN` khong bat buoc, nhung nen co neu ban muon download on dinh hon tu Hugging Face.

## Trang thai runtime hien tai

Trong workspace nay, cac thanh phan sau da duoc tai:

- `CodeT5`: [Salesforce--codet5-base](</c:/Users/Admin/Documents/1. UET/lab/VulGuardVN/GRACE-improve/baseline/artifacts/models/retrieval/Salesforce--codet5-base>)
- `Joern`: [joern-cli](</c:/Users/Admin/Documents/1. UET/lab/VulGuardVN/GRACE-improve/baseline/artifacts/graphs/tools/joern/joern-cli>)
- `Joern parse wrapper`: [joern-parse.bat](</c:/Users/Admin/Documents/1. UET/lab/VulGuardVN/GRACE-improve/baseline/artifacts/graphs/tools/joern/joern-cli/joern-parse.bat>)
- `Joern export wrapper`: [joern-export.bat](</c:/Users/Admin/Documents/1. UET/lab/VulGuardVN/GRACE-improve/baseline/artifacts/graphs/tools/joern/joern-cli/joern-export.bat>)

Tuc la neu ban de `GRACE_RETRIEVAL_BACKEND=auto` va `GRACE_GRAPH_BACKEND=auto`, pipeline hien tai se tu dong nhan `CodeT5 + Joern`.

## Cach chay

Chay trong repo root `VulGuardVN`.

1. Neu can ReVeal:

```bash
python GRACE-improve/baseline/00_download_reveal.py
```

Neu ReVeal duoc tai ve o dang parquet, convert them:

```bash
python GRACE-improve/baseline/00_5_prepare_reveal_parquet.py
```

2. Chuan hoa du lieu:

```bash
python GRACE-improve/baseline/01_prepare_datasets.py
```

3. Tao split:

```bash
python GRACE-improve/baseline/02_create_splits.py
```

4. Train pre-filter:

- Mo `GRACE-improve/baseline/03_train_prefilter.ipynb`
- Sua `DATASET_NAME` neu can
- `Run All`

5. Calibration:

```bash
python GRACE-improve/baseline/04_calibrate_prefilter.py
```

6. Tai retrieval model:

```bash
python GRACE-improve/baseline/download_codet5_retrieval_model.py
```

7. Tai Joern:

```bash
python GRACE-improve/baseline/download_joern.py
```

8. Build retrieval bank:

```bash
python GRACE-improve/baseline/05_build_demo_bank.py
```

9. Tai local LLM:

```bash
python GRACE-improve/baseline/download_vulnllm_r_7b.py
```

10. Chay full pipeline:

```bash
python GRACE-improve/baseline/06_run_grace_prefilter.py
```

11. Danh gia:

```bash
python GRACE-improve/baseline/07_evaluate_predictions.py
```

## Bien moi truong quan trong

- `GRACE_DATASET`
  - Gia tri hop le: `devign`, `bigvul`, `reveal`

- `GRACE_RETRIEVAL_BACKEND`
  - `auto`: uu tien `CodeT5`, fallback `TF-IDF`
  - `codet5`: bat buoc dung `CodeT5`
  - `tfidf`: bo semantic encoder, chi dung fallback retrieval

- `GRACE_GRAPH_BACKEND`
  - `auto`: uu tien `Joern`, fallback heuristic
  - `joern`: bat buoc dung `Joern`
  - `heuristic`: bo Joern, dung graph heuristic

- `GRACE_MAX_TEST_SAMPLES`
  - `None` mac dinh, nghia la chay full test set

- `GRACE_CALL_LLM_FOR_UNCERTAIN`
  - mac dinh `True`

- `GRACE_CALL_LLM_FOR_HIGH_RISK`
  - mac dinh `False`

## Vi du chay strict gan paper

Neu ban muon ep pipeline dung dung `CodeT5 + Joern`:

```bash
$env:GRACE_DATASET="devign"
$env:GRACE_RETRIEVAL_BACKEND="codet5"
$env:GRACE_GRAPH_BACKEND="joern"
python GRACE-improve/baseline/05_build_demo_bank.py
python GRACE-improve/baseline/06_run_grace_prefilter.py
```

## Artifact duoc sinh ra o dau

Tat ca artifact duoc ghi vao `GRACE-improve/baseline/artifacts/`:

- `models/`
  - prefilter model
  - retrieval model
  - local LLM

- `retrieval/`
  - `demo_bank.joblib`
  - retrieval summary

- `graphs/`
  - graph cache cho tung sample
  - local `Joern` installation

- `predictions/`
  - prediction jsonl
  - run state
  - issue log neu pipeline dung giua chung

- `metrics/`
  - evaluation summary
  - F1 / precision / recall / ROC-AUC / PR-AUC
  - timing breakdown

## Luu y van hanh

- Neu ban da co `demo_bank.joblib` cu tu schema truoc day, hay rebuild lai bang `05_build_demo_bank.py`.
- `06_run_grace_prefilter.py` se tu kiem tra schema cua prediction va demo bank.
- Retrieval hien da tro lai gan paper hon:
  - semantic shortlist bang `CodeT5`
  - lexical rerank bang `Jaccard`
  - syntactic rerank bang `AST sequence similarity`
- Prompt hien da dua `node information` va `edge information` vao local LLM thay vi chi dung `lightweight structural summary`.
- Neu `Joern` hoac `CodeT5` gap loi runtime, backend `auto` van co the fallback, nhung khi muon so sanh gan paper thi nen dat `codet5` va `joern` ro rang.
