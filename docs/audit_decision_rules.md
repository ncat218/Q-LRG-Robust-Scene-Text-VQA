# Quy tắc đọc audit và quyết định S1

Ghi ngày 07/11/2026, trước khi tiếp nhận kết quả audit A0. Quy tắc áp dụng cho batch size 1, model và môi trường đã ghim. Các con số của job 10756 và 10820 dưới đây là dữ liệu development đã có.

## Quy tắc quyết định audit

Ưu tiên kiểm tra giao tập và tính toàn vẹn dữ liệu trước khi diễn giải so sánh dự đoán.

| Kết quả quan sát | Diễn giải được phép | Hành động |
| --- | --- | --- |
| Manifest pilot train giao tập 500 ảnh theo image ID, question ID hoặc SHA-256 ảnh | Kết quả pilot bị nhiễm giao tập theo phép kiểm tra phát hiện được. | Đánh dấu kết quả pilot invalid, sửa split và chạy lại phép đánh giá thích hợp trước khi kết luận. |
| Thiếu artifact, sai hash hoặc điều kiện chạy khác nhau | Phép so sánh chưa được kiểm soát đầy đủ. | Hoàn tất hoặc sửa audit, giữ nguyên nguyên nhân ở trạng thái chưa xác định. |
| Hai lượt native khác token hoặc điểm | Chưa lặp lại ổn định trong điều kiện audit. Cần thêm bằng chứng để xác định nguyên nhân số học, trạng thái model, input hoặc cấu hình. | Đo riêng tỷ lệ đổi token, đổi điểm và delta accuracy với CI. Điều tra và lặp thêm trước khi chốt quy mô train. Dùng đối chứng cùng lượt trong các phép so sánh sau. |
| Native ổn định nhưng injection thay đổi giữa hai lượt | Có vấn đề về độ lặp lại trên đường injection trong phạm vi đã đo. | Kiểm tra cache, feature mới tính và prefill logits trước khi train lớn. |
| Mỗi đường ổn định, nhưng native khác injection | Có chênh lệch có hệ thống giữa hai đường trong phạm vi đã đo. | Điều tra feature cache, preprocessing, vị trí injection và logits, rồi sửa hoặc xác minh lại đường đối chứng. |
| Tất cả cặp hiện tại khớp và split đạt | Đường hiện tại đạt phép kiểm tra trên cohort này. Nguyên nhân chênh lệch lịch sử ở job 10820 vẫn cần đối chiếu metadata hoặc tái lập điều kiện cũ. | Tiếp tục các đối chứng S0 và chốt thiết kế, dữ liệu, storage cùng ngân sách S1 mở rộng. |

Tỷ lệ câu đổi token có đơn vị khác delta accuracy tính bằng điểm phần trăm. Hai lần lặp chỉ cung cấp một phép kiểm tra độ lặp lại. Để định lượng biến thiên chạy lại cần thêm lần lặp hoặc job độc lập, cùng với phân bố chênh lệch điểm. Mức biến thiên đó phải được so với hiệu ứng cần phát hiện và CI của hiệu ứng.

Cổng tự động hiện tại chỉ cho S1 mở rộng qua khi mọi cặp audit khớp. Kết quả có sai khác được giữ để điều tra và sửa thiết kế. Thay đổi tiêu chí chấp nhận cần được ghi lại trước lượt train lớn.

## Đọc S0 và S1 đúng cohort

| Cohort S0 | Tỷ lệ token O2 trung bình theo câu | Phần khoảng cách O1–O0 được O2 hồi phục |
| --- | --- | --- |
| 500 ảnh, 792 câu | 15,58% | 92,0% |
| 150 ảnh, 150 câu grounded tạm | 12,97% | 79,1% |

O0 giữ feature LR4, O1 dùng toàn bộ feature sạch, O2 dùng feature sạch ở vùng TextOCR. Các tỷ lệ hồi phục là point estimate. Trên 500 ảnh, trung bình tỷ lệ mask theo ảnh là 15,37%. Phải ghi cách lấy trung bình và dùng cùng cohort khi đối chiếu mask với accuracy.

Trên LR4, cosine toàn ảnh từ 0,604 của A0 lên 0,637 ở S1a (feature-only) và 0,617 ở S1b (answer + feature + residual). Cần xem riêng metric của từng loss và từng loại token.

## Đối chứng S0 tiếp theo

1. So overlap 5% và center trên cùng manifest, grounding và model. Báo cáo accuracy, mask size, ảnh hưởng tới chữ nhỏ và những grounding bị mất khi mask rỗng.
2. Thêm random control theo đúng số token của O2 trên từng ảnh, với nhiều seed mask cố định trước khi chạy. Dùng cùng mask của một ảnh cho mọi câu trên ảnh đó. Báo cáo phân bố kết quả theo seed và CI ghép cặp theo ảnh. Các lần bốc tăng số lần lặp của can thiệp trên cùng mẫu.
3. Với O3 (vùng đáp án theo nhãn), tiếp tục đối chiếu O4 (ngẫu nhiên cùng ngân sách O3) trên grounding đã duyệt, có câu nhiều từ và cần ngữ cảnh. Phép thử này đo giá trị của vùng đáp án có nhãn. Khả năng mô hình dự đoán được vùng đó từ LR4 và câu hỏi còn cần kiểm chứng.

## Biến thể S1 có trọng số chữ

Giữ kiến trúc, train IDs, epoch, lịch câu hỏi và optimizer như đối chứng S1b. Chỉ thay trọng số feature-matching loss. Với lỗi feature mỗi token l_i và mask chữ M_i, dùng

```text
w_i = 1 + (alpha - 1) * M_i
L_feature = sum_i(w_i * l_i) / sum_i(w_i)
```

Alpha bằng 1 khôi phục đúng loss trung bình theo token. Dùng TextOCR ở train_inner để tạo trọng số, giữ mọi ảnh train và dùng loss gốc cho ảnh thiếu nhãn. Ghi coverage nhãn và phiên bản projection. Khi suy luận, restorer tiếp tục nhận feature LR4 theo kiến trúc S1.

Báo cáo cosine và MSE riêng trên token chữ và ngoài chữ, với số ảnh/token thực sự có nhãn. Nhãn ở tập đánh giá chỉ phục vụ tính metric chẩn đoán. Tiếp tục báo cáo soft accuracy, regression, CLEAN và tương tác suy giảm.

Loss train giảm là động lực mở rộng thử nghiệm. Cần theo dõi kết quả val_inner theo ngân sách huấn luyện để biết tăng số bước có cải thiện tổng quát hóa hay tăng overfit.

## Ngân sách và điều kiện mở rộng

Đếm N câu train_inner từ manifest trước. Dùng công thức N * epoch * thời gian mỗi mẫu cho từng arm. Job 10820 đo khoảng 0,221 giây/mẫu cho A1, 0,226 cho A2, 0,235 cho S1b và 0,273 cho S1a. Thời gian S1a bao gồm phần dựng cache và I/O. Tách phần chi phí chia sẻ này khi cập nhật ước tính.

Nếu tạm lấy N = 31.000 và hai epoch, sáu arm train có tổng dự toán thô khoảng 24 giờ/seed. Thêm S1b có trọng số chữ đưa tổng lên khoảng 28 giờ/seed, tức khoảng 84 giờ cho ba seed trước chi phí đánh giá và các learning rate khác. Đây là ngoại suy từ pilot.

Mode full của D0/S0 có walltime 3 giờ. Mode s1 mặc định 6 giờ, còn ví dụ mở rộng dùng override 24 giờ. Walltime là giới hạn mỗi allocation. Lập kế hoạch các chunk có checkpoint dựa trên ngân sách và thời gian đo thực tế. Kiểm tra cả dung lượng feature cache trước khi duyệt full study.

Chọn learning rate trên một ngân sách development nhỏ và ngang nhau giữa các arm trước, sau đó dùng ít nhất ba seed cho cấu hình đã chọn. Chạy ablation riêng nếu muốn quy nguyên nhân cho việc tăng dữ liệu, tăng tỷ lệ CLEAN hoặc thêm phạt residual sạch. Bản mẫu mở rộng đang thay nhiều yếu tố đồng thời và đo hiệu quả của cả cấu hình đó.

S2/S3 vẫn đang hoãn theo kế hoạch hiện tại. Yêu cầu global S1 thắng mọi adaptation control là một cổng đầu tư chặt, cần phân biệt với điều kiện khoa học bắt buộc. Một phép chẩn đoán rẻ có thể dùng cùng checkpoint restorer và chỉ áp residual lên mask TextOCR để kiểm tra liệu sửa token nền đang làm mất lợi ích ở vùng chữ. Nếu thử, phải ghi rõ đây là oracle gate dùng nhãn ở đánh giá. Kết quả sẽ giúp quyết định có đáng đầu tư học gate ở S2 trước khi global restorer thắng mọi đối chứng hay không.
