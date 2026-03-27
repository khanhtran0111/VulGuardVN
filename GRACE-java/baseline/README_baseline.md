# GRACE-Java baseline (Python-first, GraphCodeBERT default)

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

## 2) Nguồn dữ liệu và split (quan trọng)

Script `01_split_dataset.py` hỗ trợ 3 chế độ nguồn dữ liệu:

- `--source auto` (mặc định): ưu tiên CWE-Bench-Java CSV nếu tồn tại
- `--source cwe-bench`: ép dùng `GRACE-java/data/CWE-Bench-Java/data`
- `--source candidates`: ưu tiên các benchmark candidate JSON/JSONL cũ

Khuyến nghị dùng rõ ràng `--source cwe-bench` để tránh trôi sang dataset không phải Java.

Ví dụ:

```bash
python 01_split_dataset.py --source cwe-bench --data-dir ../data/CWE-Bench-Java/data --out-dir ../work/splits
```

Gợi ý nhanh để smoke test:

```bash
python 01_split_dataset.py --source cwe-bench --data-dir ../data/CWE-Bench-Java/data --max-samples 200 --out-dir ../work/splits
```

Luu y: `--max-samples` cat theo thu tu CSV, co the gay lech nhan o tap nho. Dung cho smoke test, khong dung de ket luan chinh.

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

Tuỳ chọn hay dùng:

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

Neu Joern khong nam trong PATH:

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

Dry-run:

```bash
python 08_run_baseline_inference.py --input ../work/test_prompts.jsonl --output ../work/test_predictions.jsonl --provider dry_run
```

OpenAI:

```bash
set OPENAI_API_KEY=YOUR_KEY
python 08_run_baseline_inference.py --input ../work/test_prompts.jsonl --output ../work/test_predictions.jsonl --provider openai --model gpt-4.1
```

### Bước H. Evaluate

```bash
python 09_evaluate.py --input ../work/test_predictions.jsonl
```

## 4) Smoke test tối thiểu để kiểm tra GraphCodeBERT chạy được

Nếu bạn chỉ muốn xác nhận retrieval chạy trước:

```bash
python 01_split_dataset.py --source cwe-bench --data-dir ../data/CWE-Bench-Java/data --max-samples 200 --out-dir ../work/splits
python 02_build_ast_seq.py --split all
python 03_build_retrieval_index.py --input ../work/train_ast.jsonl --out-dir ../work/retrieval_smoke --batch-size 8 --max-length 256 --whiten-dim 64
python 04_retrieve_demo.py --input ../work/test_ast.jsonl --index-dir ../work/retrieval_smoke --output ../work/test_with_demo_smoke.jsonl --top-k 5
```

Neu chay thanh cong, ban se co:

- `work/retrieval_smoke/retrieval.index`
- `work/retrieval_smoke/metadata.pkl`
- `work/test_with_demo_smoke.jsonl`

## 5) Ghi chú kỹ thuật

- Retrieval encoder trong code hien tai la GraphCodeBERT theo mac dinh.
- AST similarity trong rerank co fallback khi thieu `python-Levenshtein`.
- Pipeline split la group-aware, nhung van can theo doi phan bo nhan de tranh tap test qua lech.
- Joern CLI co the khac nhe theo version; neu loi cu phap, kiem tra lai `joern-export --help`.

## 6) Checklist khi chạy thực nghiệm chính thức

Truoc khi chot ket qua benchmark:

1. Khong dung `--max-samples`.
2. Xac nhan train/val/test co ca 2 nhan 0/1.
3. Log ro model retrieval, pooling, index type, whiten dim.
4. Co dinh split protocol va prompt template de so sanh cong bang.
