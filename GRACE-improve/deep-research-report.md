# Thiết kế và đánh giá một pre-filter deep learning nhẹ để chèn vào pipeline GRACE nhằm tăng F1 và giảm tải LLM

## Tóm tắt điều hành

Bài toán phát hiện lỗ hổng ở mức hàm trong GRACE nhắm đến C/C++ (nhìn từ prompt template và mã chạy trong repo), và dùng LLM (được mô tả là GPT‑4 trong tổng quan) kết hợp cấu trúc đồ thị (AST/PDG/CFG) và in-context learning (ICL). citeturn6view0turn5view0turn39view0 Trong các benchmark phổ biến, một khảo sát tổng hợp báo cáo F1 của GRACE lần lượt là **0.651 (Devign/FFmpeg+Qemu), 0.431 (ReVeal), 0.355 (Big‑Vul)**. citeturn39view0

Mục tiêu của pre-filter “nhẹ” (binary classifier) là tạo một tầng lọc sớm (early triage) để:
- **Giảm số lần gọi LLM** (chi phí/độ trễ) bằng cách loại bỏ phần lớn hàm “rõ ràng không vulnerabile” trước khi chạy Joern/graph + prompt + LLM.
- **Tăng F1 end‑to‑end** bằng cách giảm false positive (LLM đôi khi “ảo giác”/over-flag), đồng thời kiểm soát false negative qua cơ chế ngưỡng–hiệu chỉnh xác suất.

Khuyến nghị trọng tâm (được tối ưu theo “tổng F1 + giảm tải LLM”, không chỉ accuracy):
- **Lựa chọn mô hình pre-filter số một:** *Transformer nhỏ/distilled code-LM* (ví dụ distillation từ CodeBERT/GraphCodeBERT) chạy trên token/subword, vì thường cho quality–cost tốt hơn CNN/RNN thuần, trong khi vẫn đủ nhẹ để đặt trước pipeline. citeturn10search1turn10search0turn11search3turn10search2  
- **Lựa chọn mô hình pre-filter số hai:** *Token-CNN hoặc BiLSTM* (nhanh, dễ triển khai, ít tham số) như baseline “siêu nhẹ” để đạt throughput lớn, dùng focal loss/weighted BCE vì mất cân bằng lớp nặng ở ReVeal và Big‑Vul. citeturn11search0turn17view0  
- **Chiến lược ngưỡng đề xuất:** *dual-threshold triage + calibration* — đặt ngưỡng thấp để **bảo toàn recall** (ưu tiên không bỏ sót lỗ hổng) và một vùng “uncertain” để gọi LLM; dùng temperature scaling/Platt scaling trên validation để đặt ngưỡng theo “recall mục tiêu” hoặc tối ưu F1 end-to-end. citeturn10search3turn37search3  
- **Giao thức đánh giá khuyến nghị:** so sánh A/B “GRACE gốc” vs “GRACE + pre-filter” theo **F1, recall lớp vuln, số call LLM, latency end‑to‑end**, có kiểm định McNemar/bootstrapping; đồng thời kiểm soát rủi ro label-noise, data leakage, và “context-dependence” của benchmark. citeturn11search14turn36view0turn35view0  

Các biến chưa xác định cần giả lập theo nhiều kịch bản: **mục tiêu latency** (ms/hàm) và **split chính xác trong GRACE** (repo thể hiện có train/test file cho retrieval nhưng không đủ để khẳng định tỉ lệ; nhiều bài khác dùng 8:1:1 hoặc 80/10/10). citeturn8view0turn17view0turn36view0  

## Bối cảnh và mục tiêu tối ưu hóa GRACE

GRACE được mô tả gồm ba mô-đun chính: (i) **chọn demonstration** cho ICL dựa trên độ tương tự (semantic/syntactic/lexical), (ii) **biểu diễn cấu trúc đồ thị** bằng cách kết hợp AST/PDG/CFG, và (iii) mô-đun tăng cường phát hiện (prompt giàu domain + graph prompt + ICL prompt). citeturn5view0

Trong mã repo GRACE, phần “prompt khung” thể hiện rõ nhiệm vụ **function vulnerability detection cho C/C++**, và chừa “slot” để thêm node/edge information của đồ thị vào prompt. citeturn6view0 Ở README, nhóm tác giả cũng nhấn mạnh phụ thuộc phiên bản Joern: **đồ thị sinh ra bởi các phiên bản Joern khác nhau có thể khác đáng kể**, kéo theo khác biệt hiệu năng. citeturn12view0  
Điều này tạo ra hai hệ quả trực tiếp cho một pre-filter “nhẹ”:
1. Nếu pre-filter cần graph/CPG từ Joern, bạn có thể **không tiết kiệm được** phần chi phí parse/graph (vì đã phải chạy Joern trước).
2. Nếu pre-filter chạy **trước Joern** (token-only/byte-level), bạn có thể tiết kiệm “toàn bộ” phần graph + retrieval + LLM cho đa số mẫu âm tính, nhưng phải chấp nhận biểu diễn đầu vào nghèo hơn.

Điểm khởi đầu thực nghiệm (baseline) mà bạn nên coi là “mốc” để đo lợi ích end‑to‑end: khảo sát báo cáo F1 của GRACE theo thứ tự Devign/FFmpeg+Qemu, ReVeal, Big‑Vul là **0.651 / 0.431 / 0.355**. citeturn39view0 Vì mục tiêu của bạn là “tăng F1 và giảm tải LLM”, pre-filter cần tối ưu theo **cascade utility** (giá trị toàn pipeline), không phải chỉ pre-filter standalone.

## Dữ liệu và các rủi ro đánh giá

### Kích thước dữ liệu, mất cân bằng lớp, và split

Bảng thống kê (được dùng trong một công trình cùng nhóm benchmark) cho ba dataset như sau: citeturn17view0

| Dataset | Tổng mẫu | Vulnerable | Non-vulnerable | Tỉ lệ vulnerable |
|---|---:|---:|---:|---:|
| FFmpeg+Qemu (Devign) | 22,361 | 10,067 | 12,294 | 45.02% |
| ReVeal | 18,169 | 1,664 | 16,505 | 9.16% |
| Big-Vul (Fan et al.) | 179,299 | 10,547 | 168,752 | 5.88% |

Từ các con số trên, mức mất cân bằng (Non/Vul) xấp xỉ:
- FFmpeg+Qemu: ~1.22 (gần cân bằng).
- ReVeal: ~9.9 (mất cân bằng mạnh).
- Big‑Vul: ~16.0 (mất cân bằng rất mạnh). citeturn17view0

Về chia tập, tài liệu benchmark này mô tả cách chia **train/val/test theo tỉ lệ 8:1:1**. citeturn17view0  
Trong khi đó, mã retrieval của GRACE thể hiện có các file “train_*” và “test_*” cho code/AST (phục vụ truy hồi demonstration), nhưng không đủ thông tin để khẳng định có/không validation và tỉ lệ chia. citeturn8view0

**Hàm ý thiết kế:** pre-filter phải xử lý class imbalance nghiêm trọng trên ReVeal/Big‑Vul; nếu dùng ngưỡng 0.5 mặc định sẽ dễ “sụp recall” lớp vulnerable.

### Rủi ro chất lượng nhãn, leakage và “context dependence”

Hai rủi ro lớn có thể làm sai lệch kết luận “F1 tăng” nhưng không thực sự nâng năng lực phát hiện lỗ hổng ngoài đời:

- **Context dependence:** một nghiên cứu ISSTA 2025 chỉ ra rằng trong các dataset ML4VD phổ biến, **với hơn 90% hàm, không thể quyết định vulnerable hay không chỉ từ code hàm mà không có ngữ cảnh**; họ cũng cảnh báo cách đánh giá function-level nhị phân có thể dẫn tới mô hình dựa vào tương quan giả (spurious features). citeturn35view0  
- **Label noise và data leakage:** một nghiên cứu 2025 về “code language models cho vulnerability detection” cho thấy nhiều benchmark có **độ chính xác nhãn vulnerable thấp**, đồng thời phân tích **duplicate/exact copy** và “time travel” do random split; họ cũng nêu cách tạo split (ví dụ 80/10/10, hoặc dùng “public split” của BigVul) để nghiên cứu leakage. citeturn36view0  

**Hàm ý thiết kế đánh giá:** Khi bạn thêm pre-filter (một tầng học máy nữa), nguy cơ “tối ưu theo spurious cues” càng tăng. Vì vậy, cần ít nhất:
- Báo cáo theo nhiều split (ngẫu nhiên + dedup + temporal/cross-project nếu làm được).
- Có kiểm định thống kê và phân tích lỗi định tính (false positives/false negatives theo pattern). citeturn36view0turn35view0  

## Thiết kế pre-filter nhẹ và biểu diễn đầu vào

### Pre-filter cần dự báo gì?

Mục tiêu tối ưu của pre-filter không nhất thiết là “dự đoán nhãn cuối cùng tốt nhất”, mà là **ước lượng rủi ro đủ tốt để triage**: hàm nào “đáng đưa sang GRACE/LLM”, hàm nào “gần như chắc chắn benign” để bỏ qua nhằm tiết kiệm chi phí.

Do đó, output tốt nhất là **một score xác suất đã hiệu chỉnh** (calibrated probability), vì bạn sẽ chọn ngưỡng theo ngân sách (budget) hoặc recall mục tiêu. Temperature scaling được chỉ ra là một phương pháp calibration đơn giản và hiệu quả trong nhiều bối cảnh. citeturn10search3 Platt scaling (logistic regression trên score) là một dạng calibration cổ điển cho binary. citeturn37search3

### Biểu diễn đầu vào: chọn “nhẹ” nhưng không gây nghẽn

Có 4 nhóm biểu diễn đầu vào chính (sắp theo mức “đắt”):

- **Raw token/subword từ code (không cần Joern):** phù hợp cho pre-filter đặt *trước* mọi bước nặng. Đây là lựa chọn ưu tiên nếu mục tiêu chính là giảm workload toàn pipeline.
- **Byte/char-level:** tránh phụ thuộc tokenizer; hữu ích khi code có ký tự hiếm/khác chuẩn, nhưng thường cần sequence dài hơn để học ngữ nghĩa.
- **Embedding từ code-LM (CodeBERT/CodeT5 encoder):** hiệu năng thường tốt, nhưng nếu embedding được tính bằng model lớn, nó có thể trở thành “điểm nghẽn” thay LLM (tức giảm call LLM nhưng tăng compute ở pre-filter).
- **Graph (AST/CFG/PDG/CPG) từ Joern:** giàu cấu trúc, thường giúp vulnerability detection, nhưng nếu cần chạy Joern cho mọi hàm thì đã mất lợi thế “lọc sớm”; hơn nữa, thay đổi phiên bản Joern có thể làm thay đổi graph và hiệu năng. citeturn12view0turn5view0  

### Tác động tới cơ chế demonstration selection của GRACE

Một phần quan trọng của GRACE là chọn demonstration cho ICL. Code trong repo thể hiện cách truy hồi dựa trên:
- embedding (CodeT5-based) + FAISS để lấy topK ứng viên,
- sau đó tính **lexical similarity bằng Jaccard trên token** và **syntactic similarity bằng Levenshtein ratio trên AST dạng chuỗi**, rồi trộn trọng số **0.7 (lexical) / 0.3 (syntactic)** để chọn ví dụ cuối. citeturn8view0

Nghiên cứu về ICL cũng nhất quán rằng **chất lượng ví dụ in-context ảnh hưởng mạnh đến kết quả**, và retrieval-based selection thường tốt hơn random. citeturn37search0turn37search1turn37search2  
Vì vậy, pre-filter có thể ngoài nhiệm vụ “gating LLM”, còn đóng vai trò **điều khiển retrieval/prompt**: ví dụ tăng K khi rủi ro cao, hoặc chọn pool demonstration theo “vulnerability type” mà pre-filter dự đoán.

### Bảng mô hình ứng viên cho pre-filter

Các ước lượng tham số/độ trễ dưới đây nên coi là **xấp xỉ theo cấu hình chuẩn**; vì bạn chưa cung cấp mục tiêu latency và hạ tầng, bảng tách theo kịch bản CPU/GPU. Những mục có tham chiếu paper (CodeBERT/GraphCodeBERT/CodeT5/DistilBERT) dựa trên họ mô hình code‑LM phổ biến. citeturn10search1turn10search0turn9search3turn10search2

| Nhóm mô hình | Input | Tham số (ước lượng) | Độ trễ suy luận (kịch bản) | Ưu điểm | Nhược điểm |
|---|---|---:|---|---|---|
| Token-CNN (TextCNN/1D-CNN) | token/subword | ~0.5–5M | CPU: rất thấp; GPU: rất thấp | Rất nhanh, dễ triển khai; tốt cho triage | Có thể kém bắt quan hệ dài; dễ học “spurious cues” |
| BiLSTM/GRU nhỏ | token/subword | ~1–10M | CPU: thấp; GPU: thấp | Mạnh hơn CNN cho phụ thuộc tuần tự | Khó tối ưu latency hơn CNN; vẫn hạn chế ngữ nghĩa sâu |
| Transformer nhỏ (4–6 layers) | subword | ~20–60M | CPU: trung bình; GPU: thấp–tb | Trade-off tốt giữa chất lượng và tốc độ | Cần careful length/attn để không chậm |
| Distilled CodeBERT (student 6 layers) | subword | ~40–80M | CPU: tb; GPU: thấp | Chất lượng thường cao hơn CNN/RNN; distillation giúp giữ performance citeturn11search3turn10search2 | Phụ thuộc tokenizer; training phức tạp hơn |
| GraphCodeBERT (hoặc distilled) | subword + data-flow | ~80–125M | CPU: tb–cao; GPU: tb | Tận dụng cấu trúc (data-flow) tốt citeturn10search0 | Tạo data-flow/graph có overhead; khó đặt *trước* pipeline |
| GNN trên graph rút gọn (AST/PDG) | graph | ~1–10M (core) | CPU/GPU: phụ thuộc parse Joern | Tận dụng cấu trúc, có thể phối hợp tốt với graph prompt | Nếu cần Joern cho mọi mẫu → giảm lợi ích “lọc sớm” citeturn12view0 |
| Linear/MLP trên “cheap features” | stats + token n-gram | ~<1M | CPU: rất thấp | Cực nhanh; làm baseline mạnh | Dễ overfit/leakage; không học sâu |

### Mục tiêu huấn luyện và xử lý mất cân bằng

Vì ReVeal và Big‑Vul mất cân bằng mạnh, loss function cần phản ứng với “nhiều negative dễ”. Focal loss được giới thiệu để xử lý extreme class imbalance bằng cách giảm trọng số các mẫu dễ và tập trung vào hard examples. citeturn11search0  
Trong thực tế cho vulnerability detection, bạn nên coi focal loss là một nhánh trong ablation cùng với:
- **Weighted BCE** (class weight theo inverse prevalence),
- **Focal loss (γ, α)**,
- **Re-sampling** (under/over-sampling) kết hợp calibration lại xác suất (vì sampling thay đổi base rate). citeturn36view0turn10search3  

Ngoài binary label, nếu dataset có CWE/type (Big‑Vul thường có CWE metadata trong nhiều công trình), bạn có thể thử **multi-task head** (binary vuln + coarse CWE group) để pre-filter học representation tốt hơn và giúp chọn demonstration theo type; tuy nhiên phải kiểm soát label noise và phân bố type lệch. citeturn36view0turn39view0

## Chiến lược tích hợp vào GRACE và giảm tải LLM

### Ba cách chèn pre-filter vào pipeline

1. **Early gate trước Joern + trước retrieval:** chạy pre-filter token-only; nếu “benign chắc chắn” thì kết luận luôn và bỏ qua mọi bước sau. Đây là cách duy nhất giúp tiết kiệm cả graph + retrieval + LLM.
2. **Gate sau Joern nhưng trước LLM:** chỉ giảm call LLM, không giảm parse graph; phù hợp nếu bạn đã phải có graph cho các mục đích khác.
3. **Score fusion (ensemble) với đầu ra LLM:** thay vì gate cứng, kết hợp score pre-filter và output LLM để giảm false positive/false negative; nhưng cần calibration và có nguy cơ “khó giải thích”. citeturn10search3turn37search3  

### Lưu đồ pipeline đề xuất (Mermaid)

```mermaid
flowchart TD
  A[Input: hàm C/C++] --> B[Pre-filter nhẹ (token/byte)]
  B -->|p_vul <= τ_low| C[Output: Non-vulnerable (skip GRACE)]
  B -->|τ_low < p_vul < τ_high| D[Uncertain pool]
  B -->|p_vul >= τ_high| E[High-risk pool]

  D --> F[Demo retrieval (CodeT5 + lexical/syntactic)]
  E --> F

  F --> G[Graph extraction (Joern) + AST/PDG/CFG prompt]
  G --> H[LLM inference (GRACE prompt)]
  H --> I[Final label + optional rationale]

  subgraph Calibration
    J[Temp/Platt scaling on val] --> B
  end
```

Lưu đồ trên phản ánh triage hai ngưỡng:
- **τ_low**: ngưỡng loại bỏ mạnh (chỉ loại khi rất chắc benign).
- **τ_high**: ngưỡng “high-risk” để áp dụng prompt mạnh hơn (nhiều demo hơn, nhiều graph info hơn), chứ không nhất thiết “kết luận vuln luôn”.

### Ngưỡng và selective prompting

Một cách thực dụng để giảm workload LLM mà không đập vỡ recall:
- Chọn **τ_low** sao cho **recall pre-filter trên lớp vulnerable ≥ R\_target** trên validation (ví dụ 0.98–0.995), rồi đo xem phần trăm mẫu rơi vào “skip LLM” là bao nhiêu.
- Với các mẫu “high-risk” (≥τ_high), tăng budget prompt: tăng K demo, hoặc đưa thêm node/edge info, vì ICL quality phụ thuộc selection mạnh. citeturn37search0turn37search1turn37search2turn8view0turn5view0  

Trong code retrieval của repo GRACE, việc trộn lexical/syntactic tương tự (0.7/0.3) là một heuristic cố định. citeturn8view0 Pre-filter có thể cung cấp tín hiệu để *điều chỉnh heuristic* theo dataset:
- Trên Big‑Vul (mất cân bằng nặng), ưu tiên giảm false positive: có thể tăng K và tăng trọng số “syntactic” cho nhóm high-risk để demo sát cấu trúc hơn.
- Trên FFmpeg+Qemu gần cân bằng, strategy có thể “thoáng” hơn.

### Calibration và “cascade mindset”

Kiến trúc cascade trong thị giác máy tính (Viola–Jones) cho thấy lợi ích của việc dùng tầng nhẹ để loại phần lớn negative, rồi dành compute cho phần còn lại. citeturn11search2 Với GRACE, tầng “compute nặng nhất” là LLM call; do đó cascade triage là tự nhiên.

Nhưng để cascade hoạt động ổn định, **calibration** là yếu tố then chốt: temperature scaling thường được dùng như một bước hậu xử lý đơn giản để hiệu chỉnh confidence. citeturn10search3

## Kế hoạch thí nghiệm, ablation, kiểm định thống kê và lộ trình triển khai

### Ma trận thí nghiệm yêu cầu

Bảng dưới nhằm đáp ứng yêu cầu “A/B so sánh GRACE vs GRACE+pre-filter” và bao phủ 3 dataset.

| Trục | Nội dung |
|---|---|
| Datasets | FFmpeg+Qemu (Devign), ReVeal, Big‑Vul citeturn17view0turn39view0 |
| Baseline | GRACE gốc (không pre-filter) — mốc F1 theo khảo sát: 0.651/0.431/0.355 citeturn39view0 |
| Pre-filter candidates | Token-CNN; BiLSTM; Transformer nhỏ; Distilled CodeBERT; (tùy chọn) GNN sau-Joern |
| Metrics chính | Precision/Recall/F1 (lớp vuln); macro/micro F1; AUC-ROC/AUC-PR; FPR@Recall; #LLM calls; latency end-to-end; peak memory; FLOPs ước lượng |
| So sánh end-to-end | (i) F1 tổng pipeline, (ii) giảm % LLM calls, (iii) giảm latency, (iv) recall giảm bao nhiêu |
| Ablations | input length; class-weight vs focal; calibration on/off; τ_low/τ_high; K demo; có/không graph prompt; distillation on/off |

### Grid hyperparameter đề xuất

| Nhóm | Tham số | Giá trị gợi ý |
|---|---|---|
| Input | max_len | 256 / 512 / 1024 (tùy latency) |
| Loss | weighted BCE | pos_weight ∈ {1, 3, 5, 10, 15} (dataset-dependent) |
| Loss | focal loss | γ ∈ {1, 2, 3}, α ∈ {0.25, 0.5, 0.75} citeturn11search0 |
| Optim | LR | 1e‑5 / 3e‑5 / 1e‑4 (Transformer); 1e‑3 (CNN/RNN) |
| Threshold | τ_low | theo recall target: {0.02–0.20} sweep |
| Threshold | τ_high | {0.5, 0.7, 0.9} |
| Calibration | temperature T | fit trên validation citeturn10search3 |
| Retrieval budget | K demo | {0, 2, 4, 8} theo risk band citeturn8view0turn37search0 |

### Kiểm định ý nghĩa thống kê

Vì bạn so sánh **hai hệ thống trên cùng tập test** (paired predictions), McNemar’s test là một lựa chọn tiêu chuẩn để kiểm tra khác biệt có ý nghĩa giữa hai classifier nhị phân trên dữ liệu ghép cặp. citeturn11search14turn11search22  
Ngoài ra, với metric như F1 (không tuyến tính), nên dùng **bootstrap CI** (resample theo mẫu) cho chênh lệch F1 và chênh lệch recall lớp vulnerable.

### Template trực quan hóa cần tạo từ kết quả thực nghiệm

Bạn có thể sinh đồ thị sau khi có output thực nghiệm (không nên vẽ bằng số giả):

- **ROC / PR curve** cho pre-filter (standalone) và cho pipeline end-to-end (nếu bạn có score).  
- **F1 vs threshold (τ_low)** và **LLM calls vs threshold**, để chọn điểm vận hành tối ưu.
- **Bar chart** so sánh (F1, Recall_vuln, #LLM calls, latency) giữa GRACE vs GRACE+pre-filter trên từng dataset.

### Lộ trình triển khai theo mốc

- **Mốc chuẩn hóa dữ liệu**
  - Khôi phục đúng 3 dataset, thống nhất schema (code, label, metadata như AST/CFG/PDG nếu có).
  - Kiểm tra dedup cơ bản (exact copy) và ghi rõ split strategy (random vs temporal/public split). citeturn36view0  

- **Mốc baseline và đo “chi phí thật”**
  - Chạy GRACE gốc để lấy: F1, recall/precision, thời gian Joern, thời gian retrieval, thời gian LLM.
  - Ghi rõ phiên bản Joern (vì nhạy). citeturn12view0  

- **Mốc pre-filter v1**
  - Huấn luyện Token-CNN và/hoặc Transformer nhỏ trên split 8:1:1 (hoặc split bạn chọn), với focal/weighted BCE. citeturn17view0turn11search0  
  - Fit calibration (temperature scaling). citeturn10search3  

- **Mốc tích hợp cascade**
  - Cài dual-threshold; thiết kế routing: skip / uncertain / high-risk.
  - Với high-risk: tăng K demo và/hoặc tăng độ giàu graph prompt (budgeted prompting). citeturn5view0turn8view0turn37search0  

- **Mốc đánh giá A/B + ablation**
  - Báo cáo theo dataset; báo cáo giảm LLM calls và latency.
  - McNemar + bootstrap CI. citeturn11search14turn11search22  

### Rủi ro và giảm thiểu

- **Rủi ro 1: “Tối ưu benchmark nhưng không tối ưu thực tế”** do context dependence và nhãn nhiễu. citeturn35view0turn36view0  
  Giảm thiểu: thêm phân tích error theo nhóm; thử temporal split; báo cáo cả AUC-PR và FPR@Recall theo mức recall mục tiêu.

- **Rủi ro 2: False negative tăng khi skip LLM**  
  Giảm thiểu: đặt τ_low theo recall target; dùng calibration; theo dõi riêng recall lớp vulnerable. citeturn10search3  

- **Rủi ro 3: Pre-filter trở thành bottleneck**  
  Giảm thiểu: ưu tiên kiến trúc “token-only nhỏ”; tối ưu max_len; quantization nếu cần.

## Khuyến nghị cuối cùng

### Hai lựa chọn pre-filter nên ưu tiên thử trước

- **Ưu tiên một: Distilled CodeBERT-style transformer nhỏ (6 layers)** làm pre-filter token-only, vì có cơ sở từ họ code-LM (CodeBERT/GraphCodeBERT) và distillation giúp đạt hiệu năng với chi phí thấp hơn mô hình gốc. citeturn10search1turn10search0turn11search3turn10search2  
- **Ưu tiên hai: Token-CNN (siêu nhẹ) + focal loss + calibration**, như v1 nhanh để thiết lập đường baseline về “giảm LLM calls mà giữ recall”. citeturn11search0turn10search3  

### Chiến lược ngưỡng vận hành đề xuất

- Fit calibration (temperature scaling) trên validation. citeturn10search3  
- Chọn τ_low để đạt **recall_vuln ≥ 0.99** trên validation; sau đó sweep τ_high để tối ưu “F1 end-to-end vs cost”.  
- Áp dụng **selective prompting**: tăng K demo cho nhóm high-risk; dựa trên bằng chứng rằng retrieval demo tốt hơn random trong ICL và GRACE có sẵn cơ chế retrieval dựa CodeT5 + lexical/syntactic. citeturn37search0turn37search1turn8view0  

### Giao thức đánh giá tối thiểu để chốt kết luận

Trên mỗi dataset (và tốt nhất là thêm một split “ít leakage”):
- Báo cáo (Precision, Recall, F1) lớp vulnerable, macro/micro F1; AUC‑PR; FPR@Recall.
- Báo cáo **% giảm LLM calls**, **latency end‑to‑end**, breakdown thời gian (pre-filter, retrieval, Joern, LLM).
- Kiểm định McNemar và bootstrap CI cho chênh lệch F1/recall. citeturn11search14turn11search22  
- Ghi chú hạn chế benchmark (context dependence/label noise) như một phần “threats to validity”. citeturn35view0turn36view0