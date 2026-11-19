# So sánh Pre/S1b và phân tích đúng sai, job 52963

Ngày kiểm tra 19/11/2026. Job 48633 và 52963 đều COMPLETED, exit code 0:0.
Các artifact trong COMPLETE.json của job 52963 đã được kiểm tra checksum.

## Điều kiện so sánh

S1b sau Merger dùng checkpoint của job 11305. Pre trước Merger dùng checkpoint
của job 48633. Pre được huấn luyện bằng mã loss và optimizer của 11305, cùng
manifest 19.758 ảnh / 31.129 câu, hai epoch / 62.258 lượt mẫu, 15.565 bước
cập nhật, seed 42, AdamW, learning rate 2e-4 và tích lũy gradient 4 mẫu.
Tỷ lệ chọn ảnh CLEAN mỗi epoch là 40%. Cả hai đóng băng encoder, Merger và LLM.
S1b có 1.029.152 tham số và Pre có 1.027.482, chênh khoảng 0,16%.
Pre dùng bottleneck 318 so với 224 của S1b để khớp ngân sách tham số.
Mục tiêu feature của cả hai được đo sau Merger. Đây là một seed.

## Cùng 5.000 câu / 3.166 ảnh official validation

| Cấu hình | CLEAN | LR4 |
| --- | ---: | ---: |
| A0, Qwen gốc | 78,992% | 63,692% |
| S1b, phục hồi sau Merger | 80,026% | 64,958% |
| Pre, phục hồi trước Merger | 80,136% | 64,234% |
| A2, LoRA attention LLM | 80,814% | 65,274% |

Pre trừ S1b trên LR4 là -0,724 điểm, CI 95% [-1,301; -0,134].
Trên CLEAN là +0,110 điểm [-0,374; +0,604]. S1b trừ A0 trên LR4 là
+1,266 điểm [+0,474; +2,064]. Pre trừ A0 là +0,542 điểm
[-0,244; +1,301]. CI lấy 10.000 mẫu lặp theo cụm ảnh, có trọng số theo câu.
Đây là các CI cho checkpoint ở seed hiện tại, chưa đo biến thiên giữa seed.

## Chuyển đổi dự đoán trên LR4

| Nhóm | Số câu |
| --- | ---: |
| Cả S1b và Pre cùng đạt điểm 1 | 2.805 |
| S1b đạt 1, Pre thấp hơn 1 | 133 |
| Pre đạt 1, S1b thấp hơn 1 | 89 |
| S1b đạt 1, Pre đạt 0 | 98 |
| Pre đạt 1, S1b đạt 0 | 62 |
| Cả hai cùng đạt 0 | 1.431 |
| Pre tăng điểm so với S1b, kể cả điểm một phần | 143 |
| Pre giảm điểm so với S1b, kể cả điểm một phần | 191 |
| Điểm bằng nhau | 4.666 |

Các hàng là những nhóm có thể chồng nhau. Không cộng toàn bộ cột để lấy
tổng câu. Có 548 câu giữ nguyên điểm nhưng khác đáp án đã chuẩn hóa.

Trong 98 ca S1b đạt 1 còn Pre đạt 0, 18 ca Pre trả lời nhiều từ hơn S1b,
54 ca cả hai đáp án chỉ có một từ. Năm ca câu trả lời Pre dài hơn và chứa
toàn bộ từ trong đáp án S1b. Những con số này là dấu hiệu để chọn ca xem,
chưa phải phân loại nguyên nhân lỗi bằng ảnh.

## Sáu ca đã xem ảnh CLEAN/LR4 và đối chiếu đáp án

| Câu | Đáp án tham chiếu chính | S1b trên LR4 | Pre trên LR4 | Quan sát |
| --- | --- | --- | --- | --- |
| 37947 | r275ulo | r275ulo, 1 | r275 ul0, 0 | Khác ký tự cuối O/0 và khoảng trắng so với đáp án tham chiếu. |
| 36702 | enter | enter, 1 | esc, 0 | Pre trả tên một phím khác. Trên CLEAN cả S1b và Pre đều trả enter. Vùng phím trên LR4 khó đọc hơn. |
| 37614 | pizza palace | pizza palace, 1 | drive-in pizza palace, 0 | DRIVE-IN có thật trên biển. Pre lấy thêm dòng phía trên, trong khi đáp án chuẩn chỉ lấy tên. |
| 37183 | golden dawn | golden pawn, 0 | golden dawn, 1 | Pre giữ đúng ký tự D trong tên. Trên CLEAN cả hai đều trả đúng. |
| 35144 | tocca | toccas, 0 | tocca, 1 | S1b thêm s. Lỗi cũng xuất hiện trên CLEAN, nên ca này không đặc thù với suy giảm LR4. |
| 36855 | barbieri | aperol, 0 | barbieri, 1 | Trên ảnh có cả APEROL lớn và BARBIERI nhỏ phía dưới. Pre trả đúng dòng dưới theo câu hỏi. Trên CLEAN cả hai đều trả barbieri. |

Sáu ca này được chọn có chủ đích để minh họa các loại khác biệt và chưa đại
diện cho phân bố của toàn tập. Việc nhìn ảnh và dự đoán cho phép mô tả lỗi
đầu ra. Nguyên nhân bên trong module cần thêm can thiệp hoặc phép đo riêng.
Đặc biệt, câu 37614 có thể mất điểm do phạm vi đáp án dù model đọc được chữ
có thật trên ảnh. Cần tách vấn đề này khi xem các ca nghi thêm chữ.

## Kết luận hiện tại

S1b sau Merger tốt hơn Pre trước Merger trên LR4 trong phép so hiện tại.
Cả hai có mức hồi phục và mức hỏng đáp án riêng. Các ca Pre làm tốt hơn
gồm sửa ký tự và chọn đúng dòng chữ nhỏ theo câu hỏi. S1b bảo toàn nhiều
câu đúng hơn trên tổng thể LR4. Chênh lệch CLEAN chưa phân giải được.
Kết quả còn giới hạn ở một seed, một kiến trúc residual và hai epoch.

## Xem và tái kiểm tra

Thư mục gốc trên HPC là
`/datastore/dungmt/textvqa/Tuan/Q-LRG-crop/runs/final-tooling-xszFH1L0/runs/final-official_val-5zxrc8y1/`.

- `review_lr4.html` cho phép chọn S1b làm mốc, Pre làm nhánh so sánh rồi lọc
  tăng điểm, giảm điểm hoặc tìm question ID.
- `error_cases.csv` chứa các cặp dự đoán và cột lý do để tiếp tục review thủ công.
- `demo.html` chứa một tập ví dụ có ảnh đầu vào và dự đoán cả bốn cấu hình.
- `eda_contact.png` ghép ảnh của sáu ca đã kiểm tra.
- `eda-review.tar.gz` đóng gói các bảng, dự đoán, manifest, HTML và ảnh review.

Official validation đã được dùng trong các thực nghiệm khác của dự án.
Lượt này so các checkpoint cố định trên cùng tập, cần giữ ghi chú này khi
báo cáo về tính tổng quát hóa.
