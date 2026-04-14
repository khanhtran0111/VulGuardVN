# Phân tích nhu cầu và bối cảnh  
Pipeline hiện tại của bạn sử dụng một lớp CNN làm **prefilter** trước khi lấy embedding (CodeT5) và chuyển mã sang đồ thị (Joern) để đưa vào LLM. Mục tiêu là nâng **độ F1**, giảm **false positive**, tăng **precision** và **accuracy**. Bạn đề xuất thay thế CodeT5 bằng UniXcoder – một mô hình đa năng hỗ trợ ngôn ngữ C/C++ – và cải tiến tầng prefilter. Nghiên cứu cho thấy UniXcoder sử dụng **AST và chú thích** để cải thiện biểu diễn mã【51†L61-L68】, và đã được tinh chỉnh trên Devign với kết quả 68.34% accuracy, F1=62.14%【49†L64-L68】. Trong khi đó, CodeT5 (một encoder-decoder kiểu T5) đã từng được ghi nhận “**đánh bại nhiều LLM trước đó**” như CodeBERT, GraphCodeBERT, GPT-2, PLBART… trên các tác vụ code【23†L139-L146】. Cần xem xét kỹ ưu nhược của cả hai: UniXcoder nổi trội ở việc khai thác cấu trúc và comment, đặc biệt dành cho C/C++; CodeT5 có lợi thế về ngữ cảnh song không dùng AST trực tiếp.  

## So sánh CodeT5 và UniXcoder  
- **Kiến trúc:** CodeT5 là encoder-decoder (T5-based) xử lý mã như chuỗi. UniXcoder là mô hình **cross-modal** (encoder-decoder với cơ chế mask-attention), tích hợp cả AST và bình luận vào quá trình huấn luyện【51†L61-L68】. UniXcoder chuyển AST thành chuỗi tuần tự để giữ toàn bộ thông tin cấu trúc【51†L61-L68】.  
- **Dữ liệu huấn luyện:** CodeT5 được huấn luyện trên CodeXGLUE (code+NL) và nhiều tác vụ code generation/understanding【23†L125-L134】. UniXcoder được huấn luyện kết hợp code và comment, và tối ưu cho cả hiểu/lập mã (bao gồm đối chiếu đa ngôn ngữ)【51†L61-L68】. Cả hai hỗ trợ nhiều ngôn ngữ, nhưng UniXcoder rõ ràng hướng đến ngôn ngữ lập trình (C/C++, v.v.).  
- **Tokenization và chiều dài:** Cả hai dùng tokenizer dạng SentencePiece/BPE (UniXcoder dùng của Microsoft). CodeT5 và UniXcoder đều giới hạn input vào khoảng 512 tokens (do padding/truncation) trong ví dụ tinh chỉnh. UniXcoder còn có thể tái cấu trúc AST thành chuỗi, giúp mở rộng thông tin.  
- **Kết quả trên Devign:** Mô hình UniXcoder tinh chỉnh cho Devign đạt F1≈62.14%, precision=69.18%, recall=56.40%【49†L167-L174】. Con số recall tương đối thấp chứng tỏ UniXcoder báo ít nhầm, nhưng bỏ sót một số trường hợp. (So sánh: CodeT5-base chưa có kết quả Devign công bố tương đương, nhưng trong nghiên cứu CodeT5 vượt hẳn các mô hình khác về nhiều tác vụ code【23†L139-L146】). 
- **Kỳ vọng:** Thay CodeT5 bằng UniXcoder trong retrieval (tạo embedding mã) hoặc thậm chí cho prefilter có thể cải thiện khả năng hiểu cấu trúc mã (giảm false positives). UniXcoder được thiết kế để mã và AST tương tác, nên khả năng tìm ví dụ tương tự và mô tả lỗ hổng có thể tốt hơn. Việc chuyển sang UniXcoder có thể làm tăng precision (giảm nhầm dương giả) nếu khai thác tốt thông tin AST, nhưng cần kiểm tra xem recall có giảm không.  

## Cải tiến tầng DL của Prefilter (Token-CNN)  
Hiện tại bạn dùng mô hình CNN với tham số embedding=128, filters=128, dense=128, dropout=0.2. Để tăng độ chính xác (precision) và F1, có thể xem xét:  
- **Tăng độ phức tạp:** Thử tăng **embedding_dim** (ví dụ 256) để biểu diễn mã tốt hơn. Tăng số filters (256, 512) giúp nắm thêm mẫu. Tuy nhiên cần cân đối với khả năng overfit.  
- **Kiến trúc khác:** Nếu CNN đơn giản chưa đủ, thử nghiệm mô hình khác như Bi-LSTM hoặc Transformer nhỏ. Ví dụ, một lớp Bi-LSTM sau embedding có thể bắt chuỗi cú pháp tốt hơn. Hoặc dùng một **Transformer encoder** nhỏ (như CodeBERT-base) để học đại diện. Lưu ý CodeBERT gốc không hỗ trợ C/C++ tốt, nhưng CodeBERT-large đào tạo mới có thể dùng.  
- **Tăng dữ liệu huấn luyện:** Hiện Devign có ~17k hàm. Có thể tập Synthetic tạo thêm mẫu nhãn “Safe” đa dạng, hoặc biến đổi mã (đổi tên biến, thêm comment không đổi logic) để tăng độ bền. Như vậy prefilter học được đặc trưng đa dạng hơn.  
- **Điều chỉnh phân bố lớp:** Nếu true/vuln mất cân bằng (Devign ~10% dương), có thể điều chỉnh class_weight để giảm ảnh hưởng false positives.  
- **Lọc ngưỡng (threshold):** Hiện tuỳ vị trí model trả nhãn 0/1. Có thể dùng xác suất thay vì nhãn cứng, đặt ngưỡng cao hơn cho nhãn positive. Điều này có thể tăng precision nhưng đổi recall. Cần tìm điểm tối ưu (ROC).  
- **Dropout và Regularization:** Thử tăng dropout (0.3-0.5) để tránh overfit, vì CNN thường học rất tốt với bộ nhỏ. Kiểm tra qua training-validation loss để điều chỉnh.  
- **Ensemble prefilter:** Kết hợp nhiều prefilter nhỏ (ví dụ khác nhau về bộ lọc) bằng voting hoặc soft-voting. Điều này có thể giảm lỗi đơn nhất của mạng.  

## Thiết kế thí nghiệm chi tiết  
Để đánh giá cải tiến, thiết kế các thí nghiệm sau:  

1. **Biến thể Retrieval:**  
   - *Baseline:* CodeT5-base encoder embedding (như hiện tại).  
   - *UniXcoder:* Dùng `microsoft/unixcoder-base` làm encoder để tạo embedding cho hàm. Tiếp tục quy trình tìm 30 ví dụ tương tự, lọc lexical/AST như cũ. So sánh pipeline này với baseline.  
   - Đánh giá: accuracy, precision, recall, F1, AUC (theo GRACE) trên Devign test. Sử dụng stratified group-kfold để đảm bảo độ tin cậy.  

2. **Cải thiện Prefilter:**  
   - *Tăng kích thước:* thử `EMBEDDING_DIM=256`, `FILTERS=256`, `DENSE_UNITS=256`.  
   - *Kiến trúc khác:* xây thêm variant dùng BiLSTM (với các tham số tương ứng). Hoặc Transformer nhỏ (sử dụng `tensorflow.keras.layers.Transformer`).  
   - *Ensemble:* Kết hợp hai prefilter (ví dụ CNN và LSTM) – lấy dự đoán trung bình hoặc đa số trước khi đưa vào pipeline.  
   - Mỗi biến thể prefilter được huấn luyện với cùng thông số còn lại (epochs, batch, dropout) và so sánh F1, precision.

3. **Kết hợp UniXcoder và Prefilter:**  
   - Thử nghiệm pipeline hoàn chỉnh với UniXcoder retrieval và với mỗi biến thể prefilter ở trên. Tức là đánh giá sự tương tác giữa việc thay đổi retrieval và prefilter.  

4. **Đánh giá tổng hợp:**  
   - Sử dụng **5-fold cross-validation** theo devign hashes (giống baseline) để đo lường độ biến thiên. Tính trung bình F1 và khoảng tin cậy.  
   - Kiểm định thống kê (paired t-test hoặc McNemar) giữa các pipeline (trước vs sau thay UniXcoder, cải tiến prefilter) để xác định cải thiện có ý nghĩa hay không.

5. **Metrics và ablation:**  
   - Luôn báo cáo đầy đủ metric: *Accuracy, Precision (có ý nghĩa quan trọng ở đây), Recall, F1, ROC-AUC, PR-AUC*.  
   - Thực hiện ablation: ví dụ chạy UniXcoder retrieval nhưng giữ prefilter cũ, và ngược lại. Giúp xác định đóng góp của từng thành phần.  
   - Giả thiết: Dự đoán **precision** sẽ tăng đáng kể sau khi cải tiến (giảm false positives), với hy vọng duy trì hoặc tăng nhẹ F1. Nếu recall giảm, cân bằng trở lại bằng cách điều ngưỡng.

6. **Chi phí tính toán:**  
   - Huấn luyện Token-CNN với kích thước lớn hơn vẫn nhẹ (GPU phổ thông). Bi-LSTM/Transformer nhỏ đòi hỏi GPU mạnh hơn chút.  
   - Tạo embedding với UniXcoder-base (đã có 0.1B tham số【49†L186-L194】) nhanh, inference cost tương đương CodeT5.  
   - Tổng cộng dự trù vài chục GPU-giờ cho toàn bộ thí nghiệm (bao gồm cross-val và grid hyperparam).  

**Ví dụ các thử nghiệm cụ thể (Bảng 1):**

| Thí nghiệm | Prefilter (Biểu đồ)        | Retrieval Embedding | Chú thích                           |
|------------|----------------------------|---------------------|--------------------------------------|
| E0         | CNN (EM=128, Filt=128)     | CodeT5-base         | Baseline hiện tại                   |
| E1         | CNN (EM=256, Filt=256)     | CodeT5-base         | Tăng độ lớn CNN                      |
| E2         | BiLSTM (128, dropout 0.3)  | CodeT5-base         | Thay CNN bằng LSTM                   |
| E3         | CNN (128)                  | UniXcoder-base      | Thay CodeT5 bằng UniXcoder           |
| E4         | CNN (256)                  | UniXcoder-base      | UniXcoder + CNN lớn                  |
| E5         | Ensemble (CNN+LSTM)        | UniXcoder-base      | Kết hợp CNN & LSTM, UniXcoder        |

_Bảng 1: Các biến thể pipeline đề xuất, so sánh cấu hình prefilter và retrieval. Kết quả sẽ báo Accuracy, Precision, Recall, F1 tương ứng trên Devign._

## Kết quả dự kiến và biện luận  
Dựa trên nghiên cứu và số liệu hiện có, chúng ta kỳ vọng:  
- **UniXcoder retrieval** (E3, E4, E5) khả năng sẽ tăng **Precision** lên (giảm false positives) nhờ hiểu AST, duy trì F1 tương đương hoặc cao hơn baseline【49†L167-L174】【51†L61-L68】. Tại E4 và E5, kết hợp prefilter lớn hơn hoặc ensemble có thể bù đắp nếu recall hơi giảm.  
- **Tăng kích thước prefilter** (E1, E2) có thể cải thiện học sâu, nâng chung Precision và F1, nhưng dễ overfit; cần validate cẩn thận. Thử nghiệm LSTM (E2) có thể bắt pattern khác so CNN, có thể hữu dụng nếu CNN đơn giản chưa đủ.  
- **Ensemble prefilter** (E5) thường giảm sai sót đơn lẻ, nên kỳ vọng F1 ổn định cao hơn baseline.  

Để trực quan hóa, có thể sử dụng:  
- **Biểu đồ thanh** so sánh metric (F1, Precision, Recall) giữa các biến thể (ví dụ cột E0–E5).  
- **Biểu đồ đường (training curve)** cho các mô hình prefilter chính, so sánh loss/accuracy trong quá trình huấn luyện.  
- **Flowchart (Mermaid)** mô tả pipeline đề xuất (như bên dưới).  

```mermaid
flowchart LR
    CodeInput --> Prefilter[Prefilter (CNN/LSTM)]
    Prefilter -->|predict| Router{Dựa trên kết quả}
    Router -->|skip| LLM[LLM in-context]
    Router -->|uncertain| LLM
    Router -->|safe| DirectOutput[Không xử lý tiếp]
    LLM -->|predict| DirectOutput
    DirectOutput --> Final[Phân loại cuối cùng]
```

**Kết luận:** Dự án nên thử nghiệm đa dạng biến thể như trên, so sánh kỹ lưỡng các metric. Đặc biệt, UniXcoder (đã chứng tỏ hiệu quả cho C/C++【49†L64-L68】) rất hứa hẹn để thay CodeT5. Kết quả từ các thử nghiệm này sẽ cho biết rõ cấu hình nào nâng cao F1 tốt nhất mà vẫn giữ precision cao. Trên cơ sở đó, tổng hợp các cải tiến cụ thể cho pipeline.  

**Nguồn:** Tham khảo [UniXcoder (ACL 2022)]【51†L61-L68】, mô hình UniXcoder trên Devign【49†L64-L68】【49†L167-L174】, và đánh giá CodeT5 trong các khảo sát model code【23†L139-L146】, để đưa ra đề xuất trên.