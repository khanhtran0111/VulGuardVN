# GRACE-Java baseline

Mục tiêu của bộ script này là dựng baseline kiểu GRACE cho Java theo pipeline:

- retrieval: GraphCodeBERT embedding + whitening + FAISS + rerank lexical/syntactic
- graph: Joern Java CPG -> dot -> graph text
- prompting: target code + graph text + retrieved demo
- prediction/eval: phân loại Vulnerable/Non-vulnerable và đo Accuracy/Precision/Recall/F1

Toàn bộ output mặc định nằm ở `GRACE-java/work`.

## 1) Chuẩn bị môi trường

Trong thư mục `GRACE-java/baseline`:

```bash
pip install -r requirements.txt
```

Yêu cầu thêm cho graph stage:

- Đã cài Joern và gọi được `joern-parse`, `joern-export` từ command line.

còn nếu chưa cài thì:

```bash
python 00_setup_joern.py
```


## 2) Nguồn dữ liệu và split (quan trọng)

Script `01_split_dataset.py` hỗ trợ 3 chế độ nguồn dữ liệu:

- `--source auto` (mặc định): ưu tiên CWE-Bench-Java CSV nếu tồn tại
- `--source cwe-bench`: ép dùng `GRACE-java/data/CWE-Bench-Java/data`
- `--source candidates`: ưu tiên các benchmark candidate JSON/JSONL cũ


Ví dụ:

```bash
python 01_split_dataset.py --source cwe-bench --data-dir ../data/CWE-Bench-Java/data --out-dir ../work/splits
```

## 3) Chạy pipeline baseline

### Bước A. Sinh AST sequence

```bash
python 02_build_ast_seq.py --split all
```

Sinh:

- `work/train_ast.jsonl`
- `work/val_ast.jsonl`
- `work/test_ast.jsonl`

### Bước B. Build retrieval index (GraphCodeBERT mặc định)

```bash
python 03_build_retrieval_index.py --input ../work/train_ast.jsonl --out-dir ../work/retrieval
```

Mặc định quan trọng:

- `--model-name microsoft/graphcodebert-base`
- `--pooling mean`

Tùy chọn hay dùng:

- `--pooling cls`
- `--index-type ivf` khi train set lớn
- `--whiten-dim 256` (hoặc giảm cho smoke test)

### Bước C. Retrieve demo cho test/val

```bash
python 04_retrieve_demo.py --input ../work/test_ast.jsonl --index-dir ../work/retrieval --output ../work/test_with_demo.jsonl
```

### Bước D. Joern per-sample

```bash
python 05_run_joern_per_sample.py --input ../work/test_with_demo.jsonl --output ../work/test_with_joern.jsonl
```

Nếu Joern không nằm trong PATH:

- `--joern-parse <path_to_joern_parse>`
- `--joern-export <path_to_joern_export>`

### Bước E. Dot -> graph text

```bash
python 06_build_graph_text.py --input ../work/test_with_joern.jsonl --output ../work/test_graph_text.jsonl
```

### Bước F. Build prompt dataset

```bash
python 07_build_prompt_dataset.py --input ../work/test_graph_text.jsonl --output ../work/test_prompts.jsonl
```

### Bước G. Inference


Gemini (đọc từ `.env`):

```bash
python 08_run_baseline_inference.py --input ../work/test_prompts.jsonl --output ../work/test_predictions.jsonl --provider gemini
```

Gemini (ép model bằng CLI):

```bash
python 08_run_baseline_inference.py --input ../work/test_prompts.jsonl --output ../work/test_predictions.jsonl --provider gemini --model gemini-2.5-flash
```

Yêu cầu cho Gemini:

- File `.env` ở thư mục `GRACE-java` có `GEMINI_API` (bắt buộc).
- `GEMINI_MODEL` là tùy chọn; nếu không có thì script dùng mặc định `gemini-2.5-flash`.

### Bước H. Evaluate

```bash
python 09_evaluate.py --input ../work/test_predictions.jsonl
```

## 4) Từng file dùng để làm gì

Phần này giải thích nhanh vai trò của mỗi script trong baseline.

### `01_split_dataset.py`

Chức năng: Tạo train/val/test split có ý thức leakage (group-aware split theo project hoặc field nhóm).

Mục đích: Đảm bảo retrieval database và tập test không bị trộn dữ liệu gần nhau một cách ngẫu nhiên.

Đầu ra chính:

- `work/splits/train.jsonl`
- `work/splits/val.jsonl`
- `work/splits/test.jsonl`
- `work/splits/all_with_split.jsonl`

Phục vụ bước nào: Là đầu vào cho `02_build_ast_seq.py`.

### `02_build_ast_seq.py`

Chức năng: Parse Java code bằng `javalang` và sinh chuỗi AST kiểu SBT-like (`ast_seq`).

Mục đích: Bổ sung tín hiệu cú pháp cho retrieval rerank (lexical + syntactic).

Đầu ra chính:

- `work/train_ast.jsonl`
- `work/val_ast.jsonl`
- `work/test_ast.jsonl`

Phục vụ bước nào: `train_ast.jsonl` dùng cho build index ở `03`; `test_ast.jsonl` dùng cho retrieve ở `04`.

### `03_build_retrieval_index.py`

Chức năng: Encode train code thành embedding (GraphCodeBERT mặc định), whitening + normalize, sau đó build FAISS index.

Mục đích: Tạo retrieval database để tìm demo gần nhất cho mỗi sample cần dự đoán.

Đầu ra chính:

- `work/retrieval/retrieval.index`
- `work/retrieval/metadata.pkl`
- `work/retrieval/stats.json`

Phục vụ bước nào: `04_retrieve_demo.py` đọc index và metadata để truy hồi top-k.

### `04_retrieve_demo.py`

Chức năng: Encode query (val/test), tìm top-k từ FAISS, rerank, gán demo tốt nhất vào từng sample.

Mục đích: Thêm demonstration phù hợp vào input để LLM suy luận tốt hơn.

Đầu ra chính:

- `work/test_with_demo.jsonl` (hoặc file output bạn truyền)

Phục vụ bước nào: `05_run_joern_per_sample.py` và các bước prompt sau đó.

### `05_run_joern_per_sample.py`

Chức năng: Chạy Joern cho từng sample, export AST/CFG/PDG dạng dot.

Mục đích: Lấy structural context cho graph-enhanced prompting.

Đầu ra chính:

- `work/test_with_joern.jsonl`
- Thư mục dot files: `work/test_with_joern_joern/...`

Phục vụ bước nào: `06_build_graph_text.py`.

### `06_build_graph_text.py`

Chức năng: Parse dot files và serialize thành text gọn (nodes/edges).

Mục đích: Biến graph thành định dạng LLM đọc được trong prompt.

Đầu ra chính:

- `work/test_graph_text.jsonl`

Phục vụ bước nào: `07_build_prompt_dataset.py`.

### `07_build_prompt_dataset.py`

Chức năng: Ghép code + graph text + demo thành prompt hoàn chỉnh.

Mục đích: Tạo dữ liệu prompt-ready để gọi LLM.

Đầu ra chính:

- `work/test_prompts.jsonl`

Phục vụ bước nào: `08_run_baseline_inference.py`.

### `08_run_baseline_inference.py`

Chức năng: Chạy LLM inference trên prompt dataset (dry-run, OpenAI hoặc Gemini).

Mục đích: Tạo nhãn dự đoán `pred_label` để đánh giá.

Đầu ra chính:

- `work/test_predictions.jsonl`

Phục vụ bước nào: `09_evaluate.py`.

### `09_evaluate.py`

Chức năng: Tính metric phân loại nhị phân từ file predictions.

Mục đích: Tổng kết chất lượng baseline (Accuracy, Precision, Recall, F1).

Đầu ra chính: In JSON metric ra màn hình.

Phục vụ bước nào: Báo cáo kết quả và so sánh ablation.

## 5) Cac giai doan chay theo thu tu

1. Chia du lieu train/val/test theo split co y thuc leakage.
2. Sinh AST sequence cho tung mau Java.
3. Build retrieval index bang GraphCodeBERT + whitening + FAISS tren tap train.
4. Retrieve demo cho tap test/val (top-k) va rerank de chon demo tot nhat.
5. Chay Joern cho tung sample de xuat AST/CFG/PDG dang dot.
6. Chuyen dot graph thanh graph text gon (nodes/edges) de dua vao prompt.
7. Ghep prompt gom: target code + graph text + retrieved demo.
8. Chay LLM inference de du doan Vulnerable/Non-vulnerable.
9. Danh gia ket qua bang Accuracy, Precision, Recall, F1.

Tom tat luong xu ly:

- Retrieval branch: code -> GraphCodeBERT embedding -> FAISS -> rerank -> demo.
- Graph branch: code -> Joern (AST/CFG/PDG) -> graph text.
- Prediction branch: code + graph text + demo -> LLM -> prediction -> metrics.
