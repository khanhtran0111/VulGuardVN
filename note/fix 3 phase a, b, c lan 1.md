# Review và hướng sửa 3 phase đầu cho CWE-Bench-Java -> GraphCodeBERT

## Các lỗi chính trong pipeline cũ

1. **Sai commit cho mẫu vulnerable**
   - Trong `fix_info.csv`, cột `commit` là **commit chứa bản fix**, không phải buggy commit.
   - Pipeline cũ lại dùng `row["commit"]` làm `buggy_commit`, nên mẫu label=1 có thể đã là code đã vá một phần hoặc toàn phần.
   - Cách đúng: dùng `project_info.buggy_commit_id` cho mẫu vulnerable; dùng `fix_info.commit` (hoặc commit fix cuối cùng chạm vào unit đó) cho mẫu fixed.

2. **Chọn fixed commit quá thô**
   - Pipeline cũ lấy `parts[0]` từ `fix_commit_ids`, trong khi README của dataset nói fix có thể trải qua **nhiều commit tuần tự**.
   - Điều này làm nhiều function lấy sai snapshot clean.

3. **Cắt method theo line number một cách ngây thơ**
   - Dùng cùng `method_start/method_end` cho cả buggy và fixed snapshot là rất dễ sai vì line number có thể dịch chuyển sau patch.
   - Pipeline mới ưu tiên tìm method theo `signature`/`method name` rồi mới fallback sang line range.

4. **Không xử lý duplicate fix rows / multi-touch functions**
   - Cùng một function có thể xuất hiện nhiều dòng trong `fix_info.csv` do nhiều commit cùng chạm vào nó.
   - Nếu không group/dedup thì dataset bị lệch phân phối và phóng đại một số CVE/project.

5. **Dataset đầu ra quá nhỏ**
   - Artifact bạn upload cho thấy train chỉ có 72 sample và valid chỉ có 4 sample.
   - Với quy mô như vậy thì F1 thấp là rất dễ xảy ra, đặc biệt khi split theo project.

6. **`max_length=256` quá ngắn**
   - Với code Java method-level, 256 token rất dễ truncate nặng.
   - Pipeline mới chuyển mặc định lên 512 và ghi lại thống kê truncation.

## Các file mới

- `phase_a_inspect_raw_improved.py`
- `phase_b_build_function_dataset_improved.py`
- `phase_c_preprocess_graphcodebert_improved.py`

## Tư tưởng của pipeline mới

### Phase A
- Audit schema kỹ hơn.
- Thống kê coverage method/class.
- Phát hiện project có multi-fix-commit.
- Phát hiện duplicated training units.

### Phase B
- Group fix rows theo unit logic: `(project_slug, file, signature)`; nếu không có signature thì fallback class-level.
- Chọn row đại diện theo **commit fix muộn nhất chạm vào unit**.
- Lấy buggy snapshot từ `buggy_commit_id`.
- Lấy fixed snapshot từ `fix_info.commit` của row đại diện.
- Trích unit bằng heuristic theo signature/method name + brace matching.
- Nếu method fail thì fallback sang class.
- Tạo thêm `focus_code` quanh vùng thay đổi để giúp model tập trung hơn khi code quá dài.
- Split theo project nhưng giữ phân phối CWE tốt hơn bằng stratification ở mức project.

### Phase C
- Tokenize bằng `graphcodebert-base`.
- Dùng `max_length=512` mặc định.
- Lưu thêm `lengths.npy`, `token_length`, `was_truncated` để debug dữ liệu dễ hơn.

## Cách chạy đề xuất

```bash
python phase_a_inspect_raw_improved.py \
  --raw-root data/raw/CWE-Bench-Java \
  --out-dir data/artifacts/phase_a

python phase_b_build_function_dataset_improved.py \
  --raw-root data/raw/CWE-Bench-Java \
  --out-dir data/artifacts/dataset \
  --cache-dir data/cache/github_raw \
  --min-lines 5 \
  --max-lines 500 \
  --focus-context 8

python phase_c_preprocess_graphcodebert_improved.py \
  --dataset-dir data/artifacts/dataset \
  --out-dir data/artifacts/features \
  --model-name microsoft/graphcodebert-base \
  --max-length 512
```

## Kỳ vọng thực tế

Pipeline mới sẽ **không tự động biến bài toán thành dễ**, nhưng nó sửa đúng các lỗi nền tảng nhất:
- nhãn đúng snapshot hơn,
- extraction đúng function/class hơn,
- bớt duplicate nhiễu,
- bớt truncate,
- split lành mạnh hơn.

Nếu sau đó F1 vẫn thấp, bước tiếp theo nên tối ưu ở **phase D/train**:
- weighted loss / focal loss,
- early stopping theo binary F1,
- learning rate 1e-5 hoặc 2e-5,
- batch size nhỏ + grad accumulation,
- thử dùng `code` vs `focus_code` vs `full_code`.
