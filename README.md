# VulGuardVN

## Tong quan

VulGuardVN la workspace thu nghiem baseline phat hien lo hong cho C/C++ theo huong GRACE. Trong repo nay, huong chinh hien tai nam o `GRACE-improve/baseline`, voi muc tieu dua pipeline ve gan tinh than GRACE goc nhung van de cao tinh thuc dung khi chay local.

Pipeline hien tai:

`prefilter -> calibration + risk routing -> CodeT5 retrieval -> graph extraction -> local VulnLLM-R-7B -> evaluation`

Artifact trong repo da co mot run end-to-end hoan chinh tren `devign`, vi vay README nay mo ta baseline moi dua tren ket qua thuc te da sinh ra trong `GRACE-improve/baseline/artifacts`.

## Pham vi du lieu

Baseline hien tai ho tro 3 benchmark:

- `devign`
- `bigvul`
- `reveal`

Danh gia day du hien tai moi co tren `devign`.

Thong ke `devign` da duoc chuan hoa:

- Tong so mau: `27318`
- Positive: `12460`
- Negative: `14858`
- Positive ratio: `45.61%`

Split hien tai duoc tao theo `code_hash` de giam leakage:

- Train: `21854`
- Val: `2733`
- Test: `2731`

## Cau truc repo lien quan

- `GRACE-improve/`: baseline dang duoc phat trien va danh gia
- `GRACE-improve/baseline/`: toan bo script cho data prep, split, retrieval, graph, inference va evaluation
- `GRACE-improve/baseline/artifacts/`: noi luu processed data, splits, models, retrieval bank, graph cache, predictions va metrics
- `GRACE-main/`: ban reference duoc giu lai de doi chieu y tuong va kien truc

## Pipeline moi trong `GRACE-improve/baseline`

### 1. Chuan hoa du lieu va tao split

- `01_prepare_datasets.py` index metadata tu dataset raw va ghi `artifacts/processed/<dataset>/index.csv`
- `02_create_splits.py` tao `train/val/test` bang `StratifiedGroupKFold`
- Rieng `reveal` co the giu official split neu raw data cung cap san

Muc tieu cua tang nay la giu benchmark sach, han che trung lap mau tuong tu giua train va test.

### 2. Prefilter va calibration

Prefilter la tang dau tien de danh gia rui ro nhanh truoc khi goi LLM.

Trang thai artifact hien tai tren `devign`:

- Prefilter duoc luu duoi dang `ensemble_prefilter`
- Calibration dung `Platt scaling`
- Nguong routing hien tai:
  - `tau_low = 0.141512`
  - `tau_high = 0.60686`

Sau calibration, moi sample duoc chia vao 1 trong 3 nhom:

- `skip`: xac suat thap, du doan truc tiep `Non-vulnerable`
- `uncertain`: dua qua retrieval + graph + local LLM
- `high`: xac suat cao, trong run hien tai duoc du doan truc tiep `Vulnerable`

### 3. Demonstration retrieval

`05_build_demo_bank.py` tao demonstration bank tu train split.

Artifact hien tai tren `devign`:

- Tong demo: `8000`
- Negative demo: `4000`
- Positive demo: `4000`
- Semantic backend: `CodeT5`

Co che retrieval hien tai gom 2 tang:

- Semantic shortlist bang `Salesforce/codet5-base`
- Rerank bang lexical `Jaccard` tren token code va syntactic similarity tren `AST sequence`

So voi ban baseline cuc ky don gian, day la buoc dua retrieval ve gan hon voi logic mechanism-aware retrieval.

### 4. Graph context

`graphs.py` uu tien `Joern`, sau do moi fallback sang heuristic graph neu backend probe that bai.

Prompt co the nhan 3 nhom thong tin cau truc:

- `node_info`
- `edge_info`
- `ast_sequence`

Trang thai artifact hien tai:

- Graph backend requested: `auto`
- Graph backend resolved: `heuristic`
- Notice: `Joern health probe failed: Joern export returned no nodes.`

Noi cach khac, pipeline da co giao dien graph-aware, nhung run hien tai tren `devign` chua khai thac duoc Joern that su.

### 5. Local detection

`06_run_grace_prefilter.py` chay full cascade va goi local model `Virtue-AI-HUB/VulnLLM-R-7B` khi can.

Run da hoan thanh trong repo dang o che do:

- `experiment_mode = local_uncertain_only`
- Chi nhung mau `uncertain` moi goi LLM
- `skip` va `high` duoc quyet dinh truc tiep boi prefilter
- Retrieval backend trong run: `codet5`
- Graph backend thuc te trong run: `heuristic`

### 6. Evaluation

- Predictions: `GRACE-improve/baseline/artifacts/predictions/devign/grace_prefilter_predictions.jsonl`
- Evaluation summary: `GRACE-improve/baseline/artifacts/metrics/devign/evaluation_summary.json`
- Retrieval summary: `GRACE-improve/baseline/artifacts/retrieval/devign/summary.json`

README nay su dung cac artifact tren lam nguon so lieu chinh.

## Ket qua hien tai tren Devign

Ket qua duoi day duoc tong hop tu artifact danh gia hien co:

| Metric | Gia tri |
| --- | --- |
| Accuracy | `57.38%` |
| Precision | `51.81%` |
| Recall | `94.38%` |
| F1 | `66.89%` |
| ROC-AUC | `0.7067` |
| PR-AUC | `0.6587` |
| TP | `1176` |
| TN | `391` |
| FP | `1094` |
| FN | `70` |
| Bootstrap F1 mean | `0.6691` |
| Bootstrap F1 interval | `[0.6518, 0.6865]` |

Routing breakdown tren `2731` sample test:

- `skip`: `171`
- `uncertain`: `552`
- `high`: `2008`
- LLM calls: `552 / 2731 = 20.21%`
- Prefilter direct decisions: `2179 / 2731 = 79.79%`

Mot vai chi so van hanh de de hinh dung chi phi:

- Average end-to-end runtime: `3383.053 ms/sample`
- Average retrieval latency tren sample co goi LLM: `221.091 ms`
- Average graph latency tren sample co goi LLM: `14.807 ms`
- Average local LLM latency tren sample co goi LLM: `16501.231 ms`

## Diem cai tien cua baseline moi

### 1. Giam chi phi goi LLM rat manh

Ban run hien tai chi dua `20.21%` test sample qua local LLM. Day la cai tien quan trong nhat ve van hanh: pipeline khong con phu thuoc vao viec cho moi sample di qua prompt day du.

### 2. Retrieval da gan hon voi GRACE hon truoc

Thay vi chi chon demo theo token overlap don thuan, pipeline moi da co:

- semantic retrieval bang `CodeT5`
- lexical rerank bang `Jaccard`
- syntactic rerank bang `AST sequence similarity`
- demo bank can bang giua positive va negative

Dieu nay giup demonstration sat hon voi co che loi va de prompt on dinh hon.

### 3. Prompt co context cau truc thay vi chi code text

Tang graph da dua `node_info`, `edge_info` va `ast_sequence` vao prompt interface. Du Joern chua chay on dinh, thiet ke pipeline da san cho graph-aware prompting thay vi chi tom tat structure o muc rat nhe.

### 4. Data hygiene tot hon cho benchmark

Split duoc tao theo `code_hash`, ho tro official split cua `reveal`, va toan bo artifact duoc chuan hoa ve `processed/splits/models/predictions/metrics`. Day la cai tien quan trong de tranh leakage va de tai lap thuc nghiem.

### 5. Pipeline local va co kha nang resume

Run state, prediction schema, graph cache, retrieval bank va LLM cache deu da co artifact rieng. Viec resume, re-run va debug thuc dung hon ban baseline chi chay mot mach roi mat dau vet.

## Han che hien tai

### 1. Precision con thap, false positive con nhieu

Du recall rat cao (`94.38%`), precision moi dat `51.81%`, voi `1094` false positive. Day la baseline screening tot, nhung chua phu hop neu muon dung nhu bo loc cuoi cung.

### 2. Nhanh `high` dang qua hung huc

Trong run hien tai, nhanh `high` duoc auto-accept la `Vulnerable` ma khong qua LLM. Accuracy rieng cua nhanh nay chi khoang `54.48%`, nen day la nguon false positive lon nhat. Neu muon tang precision, day la diem uu tien can sua dau tien.

### 3. Joern chua vao duoc pipeline that su

Du repo da co script va artifact tai Joern, graph backend thuc te van fallback sang `heuristic`. Nghia la ket qua hien tai chua phai GRACE graph-aware day du theo AST/CFG/PDG nhu muc tieu ban dau.

### 4. End-to-end metrics moi co tren Devign

`bigvul` va `reveal` da co script prepare va split, nhung repo chua co artifact danh gia cuoi cung de bao cao ngang muc voi `devign`.

### 5. LLM van la nut that co chi phi cao trong nhanh kho

Nhanh `uncertain` van mat trung binh hon `16.5s` moi lan goi local LLM. Nghia la pipeline da giam so lan goi model, nhung khi da goi thi chi phi van lon.

## Cach chay nhanh baseline hien tai

Chay tu repo root `VulGuardVN`:

```bash
python GRACE-improve/baseline/01_prepare_datasets.py
python GRACE-improve/baseline/02_create_splits.py
```

Train prefilter trong notebook:

- Mo `GRACE-improve/baseline/03_train_prefilter.ipynb`
- Chon `DATASET_NAME`
- Run all

Sau do calibrate va build pipeline:

```bash
python GRACE-improve/baseline/04_calibrate_prefilter.py
python GRACE-improve/baseline/download_codet5_retrieval_model.py
python GRACE-improve/baseline/download_joern.py
python GRACE-improve/baseline/05_build_demo_bank.py
python GRACE-improve/baseline/download_vulnllm_r_7b.py
python GRACE-improve/baseline/06_run_grace_prefilter.py
```

Mot cau hinh gan voi run hien tai:

```powershell
$env:GRACE_DATASET="devign"
$env:GRACE_RETRIEVAL_BACKEND="codet5"
$env:GRACE_GRAPH_BACKEND="auto"
$env:GRACE_CALL_LLM_FOR_UNCERTAIN="true"
$env:GRACE_CALL_LLM_FOR_HIGH_RISK="false"
python GRACE-improve/baseline/05_build_demo_bank.py
python GRACE-improve/baseline/06_run_grace_prefilter.py
```

Neu muon tong hop lai metrics tu prediction artifact, mo notebook `GRACE-improve/baseline/07_evaluate_predictions.ipynb` hoac import utility trong `GRACE-improve/baseline/evaluate_predictions.py`.

## Ket luan ngan

Baseline moi trong `GRACE-improve` da dat duoc 3 muc tieu ro rang:

- Dua pipeline ve dang `prefilter + retrieval + graph + local LLM` thay vi mot prompt thuan tuy
- Giam rat manh ti le mau phai goi LLM
- Giu recall cao tren `devign`

Tuy nhien, pipeline hien tai van la mot baseline uu tien recall hon precision. Muon tien xa hon, 2 huong can lam tiep la sua nhanh `high` de tranh auto-accept qua som, va khoi phuc Joern de graph context khong con dung heuristic fallback.
