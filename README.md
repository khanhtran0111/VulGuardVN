# VulGuardVN

## Mục tiêu dự án

VulGuardVN xây framework phát hiện lỗ hổng cho Java theo tinh thần GRACE, với trọng tâm là **mechanism-aware retrieval**:
- Dùng **GraphCodeBERT** để biểu diễn code (semantic + data-flow awareness) cho bước retrieval.
- Kết hợp code context, graph context và demonstration để hỗ trợ bước dự đoán lỗ hổng.
- Ưu tiên đánh giá trên **benchmark có nhãn trước** để tránh kết luận yếu và giảm leakage.

## Dataset

- Dataset chính: [CWE-Bench-Java](https://huggingface.co/datasets/iris-sast/CWE-Bench-Java)
- Script tải dữ liệu: `GRACE-java/dataset_download.py`

## Cấu trúc quan trọng

- `GRACE-java/baseline/`: pipeline baseline theo từng bước (split, AST/graph, retrieval, prompt, inference, evaluate)
- `GRACE-java/work/`: các file trung gian và output chạy thực nghiệm
- `GRACE-java/data/`: dữ liệu benchmark

## Pipeline baseline (LLM-assisted)

Chạy trong thư mục `GRACE-java/baseline`:

1. `python 01_split_dataset.py`
2. `python 02_build_ast_seq.py`
3. `python 03_build_retrieval_index.py`
4. `python 04_retrieve_demo.py`
5. `python 05_run_joern_per_sample.py`
6. `python 06_build_graph_text.py`
7. `python 07_build_prompt_dataset.py`
8. `python 08_run_baseline_inference.py`
9. `python 09_evaluate.py`

Output chính thường nằm trong `GRACE-java/work/`.

## Nguyên tắc thực nghiệm bắt buộc

- Benchmark-first: đánh giá trên tập có nhãn trước, sau đó mới quét whole-project.
- Chống leakage: ưu tiên split theo project/CVE family, không random split ngây thơ ở method-level.
- Báo cáo cả retrieval metrics và downstream metrics:
  - Retrieval: Recall@K, MRR, same-CWE/same-mechanism hit rate (nếu có nhãn)
  - Detection: Accuracy, Precision, Recall, F1, per-CWE F1
