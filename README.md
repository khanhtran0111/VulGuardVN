# VulGuardVN

## Dataset

Hiện thì để chuyển sang đúng ngôn ngữ java thì dùng dataet sau: [link](https://huggingface.co/datasets/iris-sast/CWE-Bench-Java)

Để có thể tải thì có thể chạy `dataset_download.py`

## Java Pipeline (No LLM) trong GRACE-java

Phạm vi: toàn bộ pipeline chạy trong thư mục `GRACE-java`, không dùng prompt/LLM.

### Thứ tự chạy script

1. `python scripts/phase_a_inspect_raw.py`
2. `python scripts/phase_b_build_function_dataset.py`
3. `python scripts/phase_c_preprocess_graphcodebert.py`
4. `python scripts/phase_d_train_classifier.py`
5. `python scripts/phase_e_export_embeddings.py`
6. `python scripts/phase_f_evaluate.py`

### A. Mục tiêu bước hiện tại

Xây pipeline Java hoàn chỉnh cho bài toán vulnerable/non-vulnerable ở mức function/method-level, đảm bảo reproducible, tránh leakage theo project/CVE, và tái sử dụng được cho retrieval.

### B. Những file cần đọc/ghi

Input chính:
- `GRACE-java/data/raw/CWE-Bench-Java/raw_data/project_info.csv`
- `GRACE-java/data/raw/CWE-Bench-Java/raw_data/fix_info.csv`
- `GRACE-java/data/raw/CWE-Bench-Java/raw_data/build_info.csv`
- `GRACE-java/data/raw/CWE-Bench-Java/raw_data/advisory/*.json`

Output trung gian/chính:
- `GRACE-java/artifacts/phase_a/*`
- `GRACE-java/artifacts/dataset/*`
- `GRACE-java/artifacts/features/*`
- `GRACE-java/artifacts/model/*`
- `GRACE-java/artifacts/index/*`
- `GRACE-java/artifacts/eval/*`

### C. 


### D. Pseudocode/logic

#### Phase A: inspect raw data

1) Input là gì
- CSV + advisory JSON trong `raw_data`.

2) Output là gì
- `artifacts/phase_a/raw_schema_summary.json`
- `artifacts/phase_a/cwe_distribution.csv`
- `artifacts/phase_a/fix_rows_per_project.csv`

3) File script nào cần có
- `scripts/phase_a_inspect_raw.py`

4) Format dữ liệu trung gian
- JSON summary + CSV thống kê.

5) Các giả định đang dùng
- File CSV đúng encoding UTF-8 và có cột theo tài liệu dataset.

6) Cách kiểm tra đúng/sai
- So khớp số lượng project/CVE/CWE giữa `project_info.csv` và summary.
- `advisory_json_count` gần bằng số `project_slug` (chấp nhận lệch nhỏ nếu nguồn chưa đồng bộ).

#### Phase B: build function-level dataset

1) Input là gì
- `fix_info.csv` để lấy file/method/line range ở commit vulnerable.
- `project_info.csv` để lấy `fix_commit_ids` và metadata.
- GitHub raw file theo commit để cắt method code.

2) Output là gì
- `artifacts/dataset/function_dataset.jsonl`
- `artifacts/dataset/train.jsonl`, `valid.jsonl`, `test.jsonl`
- `artifacts/dataset/filtered_samples.jsonl`
- `artifacts/dataset/summary.json`

3) File script nào cần có
- `scripts/phase_b_build_function_dataset.py`

4) Format dữ liệu trung gian
- JSONL, mỗi dòng có: `sample_id, pair_id, project_slug, cve_id, cwe_id, signature, file, method, buggy_commit, fixed_commit, label, code, split`.

5) Các giả định đang dùng
- `fix_info.commit` đại diện phiên bản vulnerable.
- commit đầu trong `fix_commit_ids` đại diện bản fixed để tạo cặp.
- method line range có thể dùng lại cho cả buggy và fixed; nếu không dùng được thì loại mẫu và log.

6) Cách kiểm tra đúng/sai
- Kiểm tra mỗi `pair_id` có đủ 2 nhãn (`1` vulnerable, `0` non-vulnerable).
- Kiểm tra không leakage: một `project_slug` chỉ xuất hiện ở duy nhất một split.
- Kiểm tra `filtered_samples.jsonl` để biết mẫu bị loại do thiếu file, line range lỗi, no diff, metadata thiếu.

Rule lọc dữ liệu bẩn (đã cài):
- Bỏ mẫu thiếu method/method_start/method_end.
- Bỏ mẫu có method quá ngắn hoặc quá dài (`min_loc`, `max_loc`).
- Bỏ mẫu không tải được file buggy/fixed theo commit.
- Bỏ mẫu không cắt được theo line range.
- Bỏ mẫu buggy và fixed giống hệt nhau.

#### Phase C: tokenize + preprocess cho GraphCodeBERT

1) Input là gì
- `artifacts/dataset/{train,valid,test}.jsonl`

2) Output là gì
- `artifacts/features/{split}.input_ids.npy`
- `artifacts/features/{split}.attention_mask.npy`
- `artifacts/features/{split}.labels.npy`
- `artifacts/features/{split}.meta.jsonl`
- `artifacts/features/feature_config.json`

3) File script nào cần có
- `scripts/phase_c_preprocess_graphcodebert.py`

4) Format dữ liệu trung gian
- NumPy arrays + JSONL metadata giữ nguyên `pair_id/project_slug/cve_id/cwe_id/signature/file/method`.

5) Các giả định đang dùng
- Dùng tokenizer `microsoft/graphcodebert-base`, truncation theo `max_length`.

6) Cách kiểm tra đúng/sai
- Shape `input_ids`, `attention_mask`, `labels` cùng số dòng trên từng split.
- Metadata và labels cùng số lượng mẫu.

#### Phase D: supervised training cho binary classification

1) Input là gì
- Feature files ở Phase C.

2) Output là gì
- `artifacts/model/final_model/*`
- `artifacts/model/metrics.json`
- `artifacts/model/{valid,test}.{logits,labels,probs,preds}.npy`

3) File script nào cần có
- `scripts/phase_d_train_classifier.py`

4) Format dữ liệu trung gian
- Checkpoint Hugging Face + file NumPy dự đoán.

5) Các giả định đang dùng
- Bài toán binary classification với nhãn `1=vulnerable`, `0=non-vulnerable`.

6) Cách kiểm tra đúng/sai
- Theo dõi `f1`, `precision`, `recall`, `roc_auc` trên valid/test.
- Kiểm tra confusion matrix để phát hiện bias về một lớp.

#### Phase E: export embeddings cho retrieval

1) Input là gì
- `artifacts/model/final_model`
- `artifacts/features/{split}.input_ids.npy`, `attention_mask.npy`

2) Output là gì
- `artifacts/index/{train,valid,test}.emb.npy`
- `artifacts/index/embedding_config.json`

3) File script nào cần có
- `scripts/phase_e_export_embeddings.py`

4) Format dữ liệu trung gian
- Vector float32 theo từng sample, pooling mean trên last hidden state.

5) Các giả định đang dùng
- Mỗi embedding tương ứng đúng thứ tự metadata ở Phase C.

6) Cách kiểm tra đúng/sai
- Kiểm tra số vector = số dòng metadata của split.
- Kiểm tra chuẩn hóa và cosine retrieval chạy được không lỗi shape.

#### Phase F: evaluation + error analysis

1) Input là gì
- Dự đoán test từ Phase D.
- Embeddings test từ Phase E.
- Metadata test từ Phase C.

2) Output là gì
- `artifacts/eval/metrics_binary.json`
- `artifacts/eval/metrics_retrieval.json`

3) File script nào cần có
- `scripts/phase_f_evaluate.py`

4) Format dữ liệu trung gian
- JSON metrics.

5) Các giả định đang dùng
- Query retrieval là mẫu vulnerable (`label=1`), candidate là mẫu fixed (`label=0`).
- Positive retrieval nếu cùng `pair_id`.

6) Cách kiểm tra đúng/sai
- Binary: accuracy/precision/recall/f1/roc_auc và confusion matrix hợp lý.
- Retrieval: theo dõi `Recall@K` và `MRR`; nếu thấp, kiểm tra chất lượng pair alignment từ Phase B.

### E. Rủi ro và cách debug

- Rủi ro 1: Không tải được source theo commit (repo private, file đổi tên, path sai).
	- Debug: đọc `artifacts/dataset/filtered_samples.jsonl`, thống kê theo `reason`.
- Rủi ro 2: Line range method không còn đúng ở fixed commit.
	- Debug: tăng log cho các mẫu `cannot_slice_method`, cân nhắc parse AST hoặc map theo signature nâng cao.
- Rủi ro 3: Class imbalance giữa vulnerable/non-vulnerable theo split.
	- Debug: kiểm tra `summary.json`, thêm weighted loss hoặc undersampling.
- Rủi ro 4: Leakage ẩn qua repository gần giống nhau.
	- Debug: giữ split theo `project_slug` (đã áp dụng), có thể nâng lên split theo `cve_id` nếu cần khắt khe hơn.

### F. Kết quả mong đợi

- Có bộ dataset method-level sạch, có nhãn, có metadata đầy đủ cho truy vết (`pair_id`, `project_slug`, `cve_id`, `cwe_id`, `signature`, `file`, `method`).
- Có mô hình GraphCodeBERT fine-tuned cho binary classification.
- Có embedding index sẵn cho retrieval.
- Có metrics rõ ràng cho cả classification và retrieval để phục vụ vòng lặp cải thiện tiếp theo.