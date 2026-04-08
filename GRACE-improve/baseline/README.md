# Baseline moi cho GRACE-improve

Baseline nay di theo huong trong `deep-research-report.md`: chen mot pre-filter Token-CNN truoc buoc LLM, calibration bang Platt scaling, dual-threshold triage va selective prompting voi Gemini.

## Cau truc

- `00_download_reveal.py`: tai va giai nen ReVeal tu nguon chinh thuc.
- `00_5_prepare_reveal_parquet.py`: convert ReVeal parquet thanh `jsonl` de de doc va de pipeline dung lai.
- `01_prepare_datasets.py`: index metadata cua Devign, Big-Vul va ReVeal, khong copy them mot ban processed day du cua Big-Vul.
- `02_create_splits.py`: tao split `train/val/test`; ReVeal giu official split neu co.
- `03_train_prefilter.ipynb`: train Token-CNN pre-filter va luu model vao `baseline/artifacts/models/<dataset>/prefilter_cnn_model`.
- `04_calibrate_prefilter.py`: fit Platt scaling va tim `tau_low`, `tau_high`.
- `05_build_demo_bank.py`: build demo retrieval bank tu train split.
- `06_run_grace_prefilter_gemini.py`: chay cascade `prefilter -> retrieval -> Gemini`.
- `07_evaluate_predictions.py`: tong hop metrics, bootstrap F1 va so luong LLM calls.

## Yeu cau

- Python packages: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `python-dotenv`, `google-genai`, `requests`, `joblib`
- File `.env` tai `GRACE-improve/.env`:

```env
API_GEMINI="YOUR_GEMINI_KEY"
```

`API_GEMINI` duoc script uu tien doc truoc. Neu ban muon doi model Gemini, sua `MODEL_NAME` trong `06_run_grace_prefilter_gemini.py`.

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

6. Build retrieval bank:

```bash
python GRACE-improve/baseline/05_build_demo_bank.py
```

7. Chay baseline voi Gemini:

```bash
python GRACE-improve/baseline/06_run_grace_prefilter_gemini.py
```

8. Danh gia:

```bash
python GRACE-improve/baseline/07_evaluate_predictions.py
```

## Thiet lap dataset

Moi script dang de san bien `DATASET_NAME` o dau file. Gia tri hop le:

- `devign`
- `bigvul`
- `reveal`

Mac dinh hien tai la `devign` de de smoke test va han che chi phi Gemini. Neu chay `bigvul` hoac `reveal`, sua bien nay trong:

- `03_train_prefilter.ipynb`
- `04_calibrate_prefilter.py`
- `05_build_demo_bank.py`
- `06_run_grace_prefilter_gemini.py`
- `07_evaluate_predictions.py`

## Luu y van hanh

- `06_run_grace_prefilter_gemini.py` mac dinh `MAX_TEST_SAMPLES = 200` de tranh goi Gemini qua nhieu. Muon chay full test thi sua bien nay thanh `None`.
- `01_prepare_datasets.py` hien chi tao `index.csv` va `stats.json` trong `processed/`, khong tao `records.jsonl` nua.
- `02_create_splits.py` moi ghi ra split artifact can thiet. Voi BigVul, split file chi giu code + metadata can dung cho baseline, nen nhe hon rat nhieu so voi `MSR_data_cleaned.csv`.
- Training da duoc doi sang streaming qua `tf.data`, nen khong con doc toan bo train split vao RAM truoc khi hoc.
- Retrieval dang giu tinh than GRACE: shortlist bang semantic similarity, sau do rerank bang lexical Jaccard va syntactic skeleton similarity.
- Vi ban chua co graph/Joern artifact trong `GRACE-improve/data`, prompt LLM dung `lightweight structural summary` rut truc tiep tu source code thay cho AST/PDG/CFG that su. Neu sau nay ban bo sung graph artifact, co the mo rong prompt o `gemini_client.py`.
- Tat ca artifact duoc ghi vao `GRACE-improve/baseline/artifacts/`.
