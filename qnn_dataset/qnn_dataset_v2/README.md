# VulGuardVN - Vulnerability Detection Dataset for QNN

## Tổng quan

Dataset này được thiết kế đặc biệt cho Quantum Neural Networks (QNN) trong bài toán phát hiện lỗ hổng bảo mật Java. Pipeline xử lý chuyển đổi mã nguồn Java thành vector đặc trưng cố định (fixed-size features) thông qua phân tích cây cú pháp trừu tượng (Abstract Syntax Tree - AST).

## Cấu trúc Dataset

### Input: JSONL Format
File `dataoutput_dataset.jsonl` chứa các bản ghi với cấu trúc:
- `vul_id`: Mã định danh lỗ hổng (CVE/VUL4J ID)
- `file`: Đường dẫn file nguồn
- `method`: Tên phương thức
- `version`: Phiên bản code (vulnerable/fixed/vulnerable_other)
- `label`: Nhãn phân loại (1=vulnerable, 0=fixed)
- `code`: Snippet mã Java

### Output: Feature CSV
File `qnn_features_v2.csv` chứa 128 features + metadata:
- **Metadata columns**: id, vul_id, file, method, version, label
- **AST Statistics** (8 columns): thống kê cây cú pháp
  - `ast_n_nodes`: Số node trong AST
  - `ast_n_edges`: Số cạnh (quan hệ cha-con)
  - `ast_n_leaves`: Số lá (terminal nodes)
  - `ast_max_depth`: Độ sâu tối đa của cây
  - `ast_avg_branching`: Độ phân nhánh trung bình
  - `ast_error_nodes`: Số node lỗi khi parse
  - `ast_truncated`: Cờ báo cây bị cắt (quá MAX_NODES)
  - `parse_ok`: Cờ báo parse thành công
- **Feature vectors** (128 columns): feat_0 đến feat_127

## Kiến trúc Feature Extraction

### 1. Tổng quan kiến trúc

Feature vector 128 chiều được chia thành 2 phần:
```
[Hashed Features: 118 chiều] + [Scalar Features: 10 chiều]
```

#### Hashed Features (118 chiều)
Được chia thành 3 vùng bằng nhau:
- **Node features (39 chiều)**: Mã hóa loại node trong AST (if_statement, method_invocation, ...)
- **Edge features (39 chiều)**: Mã hóa quan hệ cha-con giữa các node
- **Path features (40 chiều)**: Mã hóa đường đi từ gốc đến lá (root-to-leaf paths)

#### Scalar Features (10 chiều)
Thống kê cấu trúc AST đã chuẩn hóa:
1. `log1p(n_nodes)`: Số node (log scale)
2. `log1p(n_edges)`: Số cạnh (log scale)
3. `log1p(n_leaves)`: Số lá (log scale)
4. `log1p(max_depth)`: Độ sâu (log scale)
5. `avg_branching`: Độ phân nhánh trung bình
6. `truncated`: Cờ báo cắt (0/1)
7. `log1p(min(error_nodes, 50))`: Số lỗi parse (clipped)
8. `parse_ok`: Cờ báo parse thành công (0/1)
9. `log1p(size_total)`: Tổng kích thước AST (log scale)
10. `1.0`: Bias term (luôn bằng 1)

### 2. Kỹ thuật Signed Hashing

**Vấn đề của feature hashing thông thường:**
Nhiều token khác nhau có thể hash vào cùng một bucket (hash collision), gây mất thông tin.

**Giải pháp - Signed Hashing:**
```python
bucket = hash(token) % dim
sign = +1 hoặc -1 (dựa trên hash thứ 2)
feature[bucket] += sign
```

**Lợi ích:**
- Giảm bias do collision: các token khác nhau cộng/trừ ngẫu nhiên thay vì luôn cộng
- Tăng khả năng phân biệt giữa các pattern khác nhau
- Phù hợp với QNN vì amplitudes có thể âm/dương

### 3. Normalization Pipeline

**Bước 1: Clip outliers**
```python
hashed = np.clip(hashed, -50.0, 50.0)
```
Lý do: Tránh một vài token xuất hiện quá nhiều lần làm chi phối toàn bộ vector

**Bước 2: Log1p transformation**
```python
hashed = sign(hashed) * log1p(abs(hashed))
```
Lý do: Giảm độ lệch (skewness) của phân phối count, làm data ổn định hơn

**Bước 3: L2 Normalization**
```python
hashed = hashed / (||hashed||_2 + 1e-8)
```
Lý do quan trọng nhất cho QNN:
- QNN hoạt động trên Bloch sphere với norm cố định
- Loại bỏ ảnh hưởng của kích thước AST (file lớn vs nhỏ)
- Tập trung vào "pattern" thay vì "magnitude"

### 4. Tại sao thiết kế này phù hợp với QNN?

#### a) Fixed-size representation
- QNN yêu cầu số qubit cố định
- AST có kích thước biến đổi -> cần chuyển thành vector cố định
- Hashing cho phép mã hóa AST bất kỳ thành 128-d vector

#### b) Normalized features
- Quantum states có norm = 1 (unit vector trên Bloch sphere)
- L2 normalization tự nhiên tương thích với quantum encoding
- Tránh saturation khi encode vào góc qubit: angle = π * normalized_value

#### c) Tách biệt structure vs content
- Hashed features (118-d): Mã hóa "nội dung" code (syntax pattern)
- Scalar features (10-d): Mã hóa "cấu trúc" code (complexity metrics)
- QNN có thể học cả 2 khía cạnh này

#### d) Robust to code variations
- Signed hashing làm features ổn định với thay đổi nhỏ trong code
- Log transformation giảm nhạy cảm với count
- Normalization đảm bảo file lớn/nhỏ đều được đối xử công bằng

## Pipeline Xử lý Toàn bộ

### Stage 1: AST Parsing
```
Java Code -> Tree-sitter Parser -> AST
```
- Sử dụng tree-sitter-java để parse code
- Wrap code snippets thành class/method để parse thành công
- Xử lý lỗi parse gracefully (parse_ok=0)

### Stage 2: Feature Hashing
```
AST -> [Node types, Edges, Paths] -> Hash buckets -> 118-d vector
```
- Traverse AST để extract node types, parent-child edges, root-to-leaf paths
- Hash mỗi pattern vào bucket tương ứng với signed hashing
- Apply normalization pipeline

### Stage 3: Append Scalar Statistics
```
118-d hashed + 10-d scalars -> 128-d feature vector
```
- Tính toán các thống kê cấu trúc từ AST
- Log transform để ổn định scale
- Concatenate với hashed features

### Stage 4: QNN Preparation (Optional)
```
128-d features -> PCA -> N-d -> Angle encoding -> QNN input
```
- Giảm chiều bằng PCA (ví dụ: 128 -> 16 qubits)
- Scale to [0, π] để encode vào góc rotation của qubit
- Quantile clipping để tránh outliers

## Cấu hình và Hyperparameters

### Feature Extraction Config
```python
DIM = 128              # Tổng số chiều feature
SCALAR_DIMS = 10       # Số chiều scalar (còn lại là hashed)
MAX_NODES = 4000       # Giới hạn số node để parse (tránh quá lớn)
MAX_PATH_LEN = 8       # Độ dài tối đa của path features
SEED = 42              # Random seed cho reproducibility

USE_EDGES = True       # Có dùng edge features không
USE_PATHS = True       # Có dùng path features không

CLIP_COUNTS = 50.0     # Ngưỡng clip cho counts
APPLY_LOG1P = True     # Có apply log1p transform không
```

### Dataset Mode
```python
MODE = "all"  # Giữ tất cả versions (vulnerable/fixed/vulnerable_other)
# MODE = "clean_binary"  # Chỉ giữ vulnerable vs fixed (cho binary classification)
```

## Thống kê Dataset

Dựa trên `summary.json`:
```
Total samples: 1578
  - Vulnerable (label=1): 409 (25.9%)
  - Fixed (label=0): 1169 (74.1%)

Parse success rate: 100% (0 parse failures)
Truncated samples: 2 (0.1%)

Average AST metrics:
  - Nodes: 152.7
  - Edges: 151.7
  - Error nodes: 0.35
```

**Nhận xét về class imbalance:**
Dataset có imbalance nhẹ (1:2.8 ratio). Khi train QNN nên:
- Dùng weighted loss hoặc focal loss
- Monitor cả precision, recall, F1 (không chỉ accuracy)
- Stratified split theo label và vul_id

## Hướng dẫn sử dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

Lưu ý: File requirements.txt đã cố định phiên bản tree-sitter để tránh xung đột API.

### 2. Chạy pipeline xử lý
Mở `data_processing.ipynb` và chạy tuần tự các cell:
- Cell 1-3: Load config và utilities
- Cell 4-5: Setup tree-sitter parser
- Cell 6-7: Define featurizer class
- Cell 8-11: Load, filter, dedup data
- Cell 12-13: Extract features và save CSV
- Cell 14: (Optional) Prepare angles cho QNN

### 3. Load dataset để training
```python
import pandas as pd
import numpy as np

# Load features
df = pd.read_csv("qnn_dataset/qnn_dataset_v2/qnn_features_v2.csv")

# Tách features và labels
feat_cols = [c for c in df.columns if c.startswith("feat_")]
X = df[feat_cols].values
y = df["label"].values

# Split by vul_id để tránh data leakage
from sklearn.model_selection import GroupShuffleSplit
groups = df["vul_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
```

### 4. Chuẩn bị cho QNN (Optional)
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_idx])
X_test = scaler.transform(X[test_idx])

# Reduce dimensions
pca = PCA(n_components=16)  # 16 qubits
Z_train = pca.fit_transform(X_train)
Z_test = pca.transform(X_test)

# Convert to angles [0, π]
def to_angles(z):
    lo, hi = np.quantile(z, [0.01, 0.99], axis=0)
    z = np.clip(z, lo, hi)
    z = (z - lo) / (hi - lo + 1e-8)
    return np.pi * z

angles_train = to_angles(Z_train)
angles_test = to_angles(Z_test)
```
