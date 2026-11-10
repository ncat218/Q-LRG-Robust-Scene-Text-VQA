# Q-LRG — Lộ trình và kết quả phục hồi latent

**Tổng hợp đến hết job 12403 ngày 10/11/2026.** Phạm vi gồm kiểm chứng Qwen gốc, thử nghiệm thay feature bằng oracle, huấn luyện restorer và các đối chứng, rồi thử crop theo câu hỏi và ghép toàn ảnh với crop.

**RQ1** hỏi liệu phục hồi visual feature có cải thiện trả lời trên LR4 hơn các cách adaptation học từ đáp án. **RQ2** hỏi việc tập trung vào chữ và vùng liên quan câu hỏi có tăng điểm trong khi vẫn giữ thông tin cần thiết từ toàn ảnh. Nhánh crop kiểm tra phần chọn đầu vào của RQ2. Gate phục hồi latent theo câu hỏi là một cơ chế riêng.

## 1. Bức tranh chung

| Câu hỏi | Kết quả hiện có |
| --- | --- |
| Khi ảnh bị giảm độ phân giải, feature ở vùng chữ có quan trọng không? | S0 thay bằng feature sạch ở khoảng 15,6% token thuộc vùng chữ và lấy lại khoảng 92% khoảng cách tới feature sạch toàn ảnh, trên 792 câu chẩn đoán. |
| Có học được cách phục hồi feature từ ảnh suy giảm không? | S1b (restorer học từ feature sạch và đáp án) tăng **1,85 điểm** so với Qwen gốc trên LR4. A2 (LoRA ở LLM) tăng **3,01 điểm** trong cùng job 11305, trên cùng 3.473 câu. |
| Crop theo câu hỏi có giữ đủ thông tin để trả lời không? | SD-RPN chọn vùng hữu ích hơn khi dùng câu hỏi đúng, nhưng dùng crop riêng làm mất chữ hoặc ngữ cảnh ở nhiều câu. Nới box 20% mỗi cạnh giúp thu hẹp phần hụt. |
| Giữ cả toàn ảnh và crop có đáng chia token không? | Trên 5.000 câu validation chính thức, ghép crop SD-RPN với toàn ảnh tốt hơn ghép crop ngẫu nhiên **1,65 điểm**. So với dùng cả ngân sách cho toàn ảnh, mức chênh là **+0,26 điểm** với CI 95% từ −0,56 đến +1,07. Cả ba nhánh dùng A2 và đúng 576 visual token. |

### Luồng dữ liệu

```text
Ảnh LR4 + câu hỏi ──→ SD-RPN ──→ box ──→ crop LR4

Toàn ảnh, crop hoặc cả hai ──→ Qwen Vision + Merger ──→ visual feature
                                                             │
                                        Giữ nguyên hoặc chỉnh bằng S1a / S1b / A1
                                                             ↓
                                     Câu hỏi + visual feature ──→ Qwen LLM ──→ đáp án
                                                                       ↑
                                                                A2 gắn LoRA
```

- **Latent** là dãy vector sau Merger và trước LLM. Restorer tính phần chỉnh sửa `R(F)` rồi đưa `F + R(F)` vào các vị trí visual token của Qwen. Ảnh hiển thị trong file review là ảnh đầu vào hoặc overlay của vị trí token. Vector đã chỉnh được dùng trực tiếp để sinh câu trả lời.
- **CLEAN** dùng ảnh gốc. **LR4** giảm mỗi chiều ảnh 4 lần rồi nội suy bicubic về kích thước ban đầu. Cặp CLEAN/LR4 vì thế có cùng hình học trước khi Qwen xử lý. Khi huấn luyện S1a/S1b, feature CLEAN cùng ảnh là mục tiêu để học từ LR4. Khi đánh giá, mỗi nhánh nhận đúng điều kiện ảnh đang chấm.
- **Crop theo câu hỏi** lấy box từ ảnh LR4 và câu hỏi, cắt ở mức pixel rồi encode crop. Nhánh crop đưa toàn bộ visual token mà Qwen tạo cho ảnh crop vào LLM sau bước resize. Nhánh ghép đưa hai dãy visual token vào hai vị trí ảnh trong cùng prompt. Số token mỗi ảnh được phân bổ trước khi encode ở các job giới hạn ngân sách.

### Chú thích các tên dùng xuyên suốt

| Tên | Chức năng |
| --- | --- |
| **D0** | Đo Qwen gốc trên CLEAN và LR4. Kiểm tra việc đưa lại feature nguyên bản vào LLM có giữ kết quả hay không. |
| **S0** | Chẩn đoán bằng oracle. Tạm thay một số feature LR4 bằng feature CLEAN đã căn hàng để đo phần thông tin có thể lấy lại. O0–O5 là các cách thay trong S0, được giải thích ở mục 3. |
| **S1** | Huấn luyện và so sánh các module trên toàn ảnh. Dùng A0 làm mốc cùng job. |
| **A0 (Qwen gốc)** | Vision Encoder và LLM giữ nguyên trọng số. Trong S1, A0 đưa lại feature nguyên bản qua cùng đường injection. |
| **S1a (restorer học khớp feature)** | Module có hai khối xử lý trên lưới không gian. Học kéo feature về feature sạch cùng ảnh bằng MSE. Cấu hình mở rộng phạt chỉnh quá mức trên ảnh sạch. |
| **S1b (restorer học khớp feature và trả lời)** | Cùng kiến trúc S1a, thêm loss dự đoán đáp án qua LLM đóng băng và phạt độ lớn phần chỉnh sửa. |
| **A1 (visual adapter học từ đáp án)** | Chỉnh từng visual token bằng adapter học từ loss trả lời. Vision Encoder và LLM đóng băng. |
| **A2 (LoRA ở LLM học từ đáp án)** | Gắn module hạng thấp vào phép chiếu Q và V của attention trong LLM. Các trọng số gốc và visual feature giữ nguyên. |
| **A1clean / A2clean** | Cùng kiến trúc A1 / A2, chỉ dùng ảnh CLEAN khi huấn luyện. Vẫn chấm riêng trên CLEAN và LR4. |
| **ROI / fallback** | ROI là box vùng quan tâm. Khi bộ chọn thiếu box hợp lệ, nhánh dùng toàn ảnh thay cho crop theo quy tắc của job. |
| **S2 / S3** | S2 dự kiến học gate để ưu tiên sửa token vùng chữ. S3 dự kiến học gate có thêm điều kiện câu hỏi. Crop SD-RPN đã chạy ở nhánh riêng và chọn box trước khi tạo visual token. |

### Tập dữ liệu và cách so sánh

| Tập | Ảnh | Câu | Vai trò |
| --- | ---: | ---: | --- |
| Smoke | 12 | 23 | Kiểm tra chạy được từ đầu tới cuối trên Slurm. |
| Tập chẩn đoán 500 ảnh | 500 | 792 | Tập con phát triển từ official train có chú thích TextOCR, dùng ở D0/S0, S1 pilot và audit A0. |
| `train_inner` | 19.758 | 31.129 | Dùng toàn bộ cho S1 mở rộng. Pilot lấy 2.048 ảnh, 3.199 câu. |
| `val_inner` | 2.195 | 3.473 | Tách từ official train theo image ID. Dùng so các checkpoint S1 và phát triển nhánh crop. |
| Tập có ROI trong `val_inner` | 2.025 | 2.996 | Chỉ các câu SD-RPN tìm được box. Dùng phân tích theo box và job ghép hai ảnh đầu tiên. |
| `official_val` | 3.166 | 5.000 | TextVQA validation chính thức. Job 12403 xác nhận ba cách phân bổ token với A2. |

Điểm là TextVQA accuracy có điểm từng phần theo mức đồng thuận đáp án. Các chênh lệch trong bảng dùng **điểm phần trăm**. CI 95% dùng bootstrap 10.000 lượt theo **image ID**, ghép dự đoán cùng câu và lấy lại mẫu theo cụm ảnh. So sánh chính giữ cùng câu hỏi, cùng điều kiện ảnh và cùng mốc đối chứng. Các job ghi **576** còn khóa tổng visual token của mỗi câu. Các job dùng độ phân giải tự nhiên ghi số token trung bình để thấy phần chi phí khác nhau.

Audit kiểm tra trùng **image ID, question ID và SHA-256 ảnh**. Tập train pilot tách khỏi 500 ảnh chẩn đoán. Với manifest mở rộng, `train_inner`, `val_inner` và `official_val` tách nhau. S1 học từ đáp án tập train. Bộ chọn crop nhận ảnh LR4 và câu hỏi. TextOCR và đáp án chuẩn dùng để dựng oracle S0, chấm điểm hoặc phân tích lỗi sau suy luận.

## 2. Trình tự các lượt chạy

Ngày trong bảng là ngày hoàn tất theo giờ HPC. Hai nhánh S1 mở rộng và crop có thời gian chạy chồng nhau.

| Ngày | Job | Việc đã làm | Dữ liệu đánh giá | Thời gian |
| --- | --- | --- | --- | --- |
| 06/11 | **10684** | Smoke D0/S0 trên H200 MIG. Sinh baseline, identity check và memo oracle. | 23 câu | 3 phút 05 |
| 06/11 | **10756** | D0 đo khoảng cách CLEAN/LR4. S0 đo vị trí nào có headroom phục hồi. | 792 câu, O3–O5 trên 150 câu | 19 phút 14 |
| 07/11 | **10820** | S1 pilot. So hai restorer với A0, A1, A2 ở quy mô nhỏ. | 792 câu | 1 giờ 15 |
| 07/11 | **11172** | Audit A0. Lặp native và injection, kiểm tra tách tập, dựng manifest đầy đủ. | 792 câu cho parity | 29 phút 10 |
| 07/11 | **11304** | Crop thử bằng CLIP. So toàn ảnh, crop theo câu hỏi, crop theo câu hỏi tráo. | 3.473 câu LR4 | 51 phút 04 |
| 08/11 | **11535** | Thay CLIP bằng SD-RPN của Danh để chọn box. | 3.473 câu LR4 | 51 phút 34 |
| 08/11 | **11841** | Dùng checkpoint S1b từ lượt train mở rộng. Bật/tắt phục hồi trên toàn ảnh và crop. | 3.473 câu LR4 | 36 phút 57 |
| 09/11 | **11920** | So A0, S1a, S1b, A1, A2 trên hai đầu vào. Dữ liệu dự đoán dùng phân tích lỗi crop. | 3.473 câu LR4 | 1 giờ 07 |
| 09/11 | **11305** | Hoàn tất S1 mở rộng, tiếp tục từ lần train đầu 11188. Đánh giá sáu checkpoint và A0. | 3.473 câu × CLEAN/LR4 | 42 giờ 17 |
| 09/11 | **12253** | Giữ A2, thử bỏ zoom, nới box 10%/20% và fallback. | 3.473 câu LR4 | 1 giờ 00 |
| 10/11 | **12289** | Ghép toàn ảnh + crop, chia 288 + 288 token. Bật S1b ở từng đầu vào và cả hai. | 2.996 câu LR4 có ROI | 57 phút 54 |
| 10/11 | **12288** | Ghép checkpoint S1b và A2, thử toàn ảnh và crop nới 20%. | 3.473 câu LR4 | 1 giờ 07 |
| 10/11 | **12376** | Mở rộng so sánh 576 token sang đủ `val_inner` và thêm toàn ảnh A2. | 3.473 câu LR4 | 1 giờ 02 |
| 10/11 | **12403** | Khóa A2, kiểm chứng toàn ảnh, ghép crop SD-RPN và ghép crop ngẫu nhiên. | 5.000 câu LR4 `official_val` | 1 giờ 11 |

### Thiết lập HPC và các lượt phải chạy lại

- Tách repo nghiên cứu Q-LRG khỏi MonkeyOCR. Launcher Python theo cách chạy của MonkeyOCR tạo script Slurm và submit ngay, sau đó khởi tạo môi trường trong batch job. Việc chuyển setup vào batch xử lý tình huống `uv` gặp giới hạn luồng trên login node. Các job dùng H200 MIG 35 GB, partition `normal`, QoS `mig2-35` và GRES `gpu:2g.35gb:1`.
- **11188 → 11305.** Lượt đầu hết quota ghi của tài khoản `dungmt`, gặp lỗi `Disk quota exceeded` khi ghi checkpoint. Phép đo dung lượng từng thư mục cho thấy feature cache chung chiếm khoảng 100 GiB trước khi dọn. Đã dọn cache tái tạo được, resume S1a từ checkpoint và dùng cache chỉ đọc. Khi cache miss, job tính feature trên GPU rồi giữ trong bộ nhớ cho ảnh hiện tại. Giới hạn ghi checkpoint được đưa về khoảng 5 mốc mỗi nhánh, cập nhật cùng file `checkpoint.pt`. Thời gian 42 giờ 17 là thời gian thực thi của job resume gồm các nhánh train nối tiếp và đánh giá.
- **11235 → 11304.** Lượt CLIP đầu dừng sau 37 giây. Đã kiểm lại file trọng số cục bộ, tải đủ khoảng 577 MiB, đối chiếu SHA-256 và dùng file đã xác minh để chạy tiếp.
- **11491 → 11535.** Lượt SD-RPN đầu dừng vì `pyexpat` thiếu symbol của thư viện Expat. Launcher sau đó nạp đúng thư viện cho môi trường selector rồi chạy lại.
- **11223** là crop oracle dựa vào vùng đáp án, hủy khi đang đợi nên không có kết quả. **11306** là lượt submit trùng của S1, đã hủy. **11908** hủy khi đang đợi và được thay bằng **11920** có thêm A2.

## 3. D0 và S0 — đo trần phục hồi trước khi train

### D0 (baseline và kiểm tra injection)

| Cùng 792 câu, 500 ảnh | Accuracy |
| --- | ---: |
| Qwen gốc, ảnh CLEAN | **84,24%** |
| Qwen gốc, ảnh LR4 | **66,29%** |
| CLEAN − LR4 | **+17,95 điểm** [14,94; 21,02] |

Kiểm tra identity lấy feature của chính ảnh đầu vào và đưa lại vào prompt. Job 10756 khớp chính xác logits và token sinh ra trên bốn grid được kiểm. Bước này kiểm đường truyền feature trước khi thay feature hoặc train module.

### S0 (oracle thay feature theo vùng)

Mọi nhánh dưới đây dùng **toàn ảnh LR4** cùng câu hỏi. Oracle chọn một số vị trí để chép feature từ ảnh CLEAN đã căn hàng. Với O3–O5, tập con gồm 150 câu trên 150 ảnh. Mỗi câu có đáp án đồng thuận ít nhất 6/10, khớp duy nhất một vùng OCR theo quy tắc tạm thời. Cohort này thiên về đáp án dễ khớp chữ và chỉ phủ 18,94% số câu.

| Nhánh | Vị trí lấy feature CLEAN | Điểm trên toàn bộ 792 câu | Điểm trên 150 câu có grounding |
| --- | --- | ---: | ---: |
| **O0 (giữ nguyên feature LR4)** | Không thay vị trí nào. Mốc của S0. | 66,20% | 71,73% |
| **O1 (thay toàn bộ feature)** | Toàn bộ visual token. Mốc gần với Qwen đọc ảnh sạch. | 84,37% | 93,73% |
| **O2 (thay ở mọi vùng chữ)** | Token có ô lưới giao với vùng TextOCR theo ngưỡng overlap 5%. | 82,92% | 89,13% |
| **O3 (thay ở vùng chứa đáp án)** | Token giao với vùng OCR đã khớp bằng đáp án chuẩn. | — | 90,20% |
| **O4 (thay ngẫu nhiên cùng số token O3)** | Vùng ngẫu nhiên với đúng ngân sách của O3. | — | 71,73% |
| **O5 (thay ở vùng đáp án đã nới)** | Vùng O3 mở thêm một ô trên lưới sau Merger. | — | 92,20% |

- **O1 so với O0** đặt trần cho phép thử. Chênh **+18,17 điểm** [15,17; 21,19] trên 792 câu, gần với khoảng cách CLEAN/LR4 của D0.
- **O2 so với O0** đo giá trị của việc có feature sạch tại chữ. Chênh **+16,72 điểm** [13,95; 19,57] với trung bình **15,58%** token được thay. Tỷ lệ thu hồi là 16,72 / 18,17, khoảng **92%**, tính trên đủ 792 câu. Trên 150 câu grounding, O2 dùng 12,97% token và lấy lại 17,40 / 22,00, khoảng **79%**.
- **O3 so với O4** kiểm vai trò của vị trí khi giữ ngân sách khoảng **2,28%** token. Chênh **+18,47 điểm** [12,53; 24,73]. O4 có một lần bốc ngẫu nhiên và cùng ngân sách O3. Đối chứng ngẫu nhiên cỡ O2 còn mở.
- **O3 so với O2** đo thêm giá trị khi chỉ dùng vùng đáp án thay cho mọi vùng chữ. Chênh **+1,07 điểm** [0,00; 2,80]. **O5 so với O3** chênh **+2,00 điểm** [0,00; 4,67] khi nới vùng. Cả hai so sánh dùng 150 câu grounding tạm. O3 dùng đáp án để định vị, nên kết quả đo headroom khi biết vùng đúng. Job crop theo query tự chọn box từ ảnh và câu hỏi ở mục 5.

## 4. S1 — từ pilot tới huấn luyện toàn bộ train_inner

### Pilot 10820

Train một epoch trên **2.048 ảnh, 3.199 câu**. Mỗi nhánh chấm cả CLEAN và LR4 trên cùng **792 câu**. Lượt này thử xem restorer có lợi ích ngoài hai loại adaptation học từ đáp án hay không.

| Nhánh | CLEAN | LR4 | Δ LR4 so với A0 |
| --- | ---: | ---: | ---: |
| **A0 (Qwen gốc)** | 84,29% | 66,09% | — |
| **S1a (khớp feature sạch)** | 82,98% | 65,72% | −0,37 |
| **S1b (khớp feature sạch + học đáp án)** | 86,49% | 67,47% | +1,39 |
| **A1 (visual adapter học đáp án)** | 86,72% | 67,70% | +1,62 |
| **A2 (LoRA ở LLM học đáp án)** | 85,33% | **68,71%** | +2,63 |

CI của S1b trên LR4 là [−0,42; +3,24]. A2 đạt [0,82; 4,47]. Lượt pilot dùng một seed và một epoch, đồng thời thấy một số câu A0 sinh token khác baseline D0. Hai việc sau đó được tách ra thành audit đường suy luận và mở rộng dữ liệu train.

### Audit 11172

Chạy hai lượt Qwen native và hai lượt injection A0 trên 792 câu. CLEAN khớp ở mọi cặp. Trên LR4, cặp native lặp lại khác tối đa 3 chuỗi token và 1 điểm câu, cặp injection lặp lại khác 2 chuỗi và 0 điểm, còn các cặp chéo native/injection khác 1–2 chuỗi và 0–1 điểm câu. Các ca được mở ra kiểm có feature và logit prefill khớp. Chênh lệch xuất hiện ở decode và nằm trong độ dao động khi lặp cùng đường. Báo cáo gốc giữ cờ `passed: false` theo luật khớp tuyệt đối. Cấu hình mở rộng dùng luật audit v2 so mức chênh giữa hai đường với mức dao động trong cùng đường, và chạy A0 cùng job để làm đối chứng.

So với D0 cũ, audit có 43/1.584 cặp câu–điều kiện khác chuỗi token, với chênh điểm tổng khoảng −0,08 điểm. Nguyên nhân dao động giữa các job chưa được xác định.

### Train mở rộng 11305

Sáu nhánh học với seed 42, **2 epoch**, learning rate `2e-4` và tích lũy gradient 4 mẫu. Các nhánh train hỗn hợp đi qua **62.258 lượt câu**, gồm 24.855 lượt CLEAN và 37.403 lượt LR4. A1clean/A2clean đi qua cùng số lượt câu, dùng ảnh CLEAN cho toàn bộ lượt train. S1a và S1b dùng cùng restorer khoảng 1,03 triệu tham số, kèm phạt residual cho ảnh sạch. Gradient từ loss trả lời đi qua Qwen đóng băng để cập nhật module cần học. Toàn bộ nhánh chấm trên cùng **3.473 câu** của `val_inner`.

| Nhánh | CLEAN | Δ CLEAN vs A0 | LR4 | Δ LR4 vs A0 |
| --- | ---: | ---: | ---: | ---: |
| **A0 (Qwen gốc)** | 83,87% | — | 65,50% | — |
| **S1a (khớp feature sạch)** | 83,41% | −0,46 | 64,64% | −0,86 |
| **S1b (khớp feature sạch + học đáp án)** | 86,15% | +2,28 | 67,36% | +1,85 |
| **A1 (visual adapter học đáp án)** | 86,13% | +2,26 | 67,71% | +2,20 |
| **A2 (LoRA ở LLM học đáp án)** | 85,97% | +2,10 | **68,51%** | **+3,01** |
| **A1clean (visual adapter train CLEAN)** | **86,54%** | +2,67 | 66,97% | +1,46 |
| **A2clean (LoRA ở LLM train CLEAN)** | 85,90% | +2,03 | 67,05% | +1,55 |

- **S1a** nâng cosine trung bình của feature LR4 so với CLEAN từ khoảng **0,609 lên 0,653** và giảm điểm LR4 **0,86**. Chỉ số gần feature sạch và điểm trả lời đang đo hai mục tiêu khác nhau.
- **S1b** cải thiện LR4 so với A0 **+1,85** [0,92; 2,79]. So với A1, chênh **−0,35** [−0,98; +0,29]. A2 hơn S1b **+1,15** [0,40; 1,89]. Hiệu ứng riêng của S1b trên LR4, tính bằng ΔLR4 − ΔCLEAN so với A0, là **−0,43** [−1,54; +0,73]. Vì vậy có bằng chứng điểm tăng so với Qwen gốc, còn lợi ích riêng do phục hồi suy giảm chưa tách rõ khỏi adaptation.
- **Train chỉ CLEAN vẫn chuyển sang LR4.** A1clean và A2clean lần lượt tăng 1,46 và 1,55 điểm so với A0 trên LR4. So trực tiếp, A2 train hỗn hợp hơn A2clean **+1,46** [0,76; 2,17] trên LR4 và **+0,07** [−0,53; +0,67] trên CLEAN. Bằng chứng chuyển miền ở đây giới hạn ở CLEAN → LR4 trên TextVQA.
- Pilot và lượt mở rộng khác quy mô train, tỷ lệ CLEAN, số epoch, mức phạt residual và tập chấm. Các chênh lệch dùng để kết luận nằm trong từng job. Bảng 11305 đánh giá `val_inner`, với một seed.

## 5. Crop theo câu hỏi — chọn vùng, thử phục hồi và phân tích lỗi

### Chọn vùng bằng CLIP rồi SD-RPN

Bộ chọn nhận LR4 và câu hỏi, xuất một box để cắt. Nhánh **query tráo** dùng câu hỏi của mẫu khác khi chọn box. Qwen ở cả hai nhánh vẫn trả lời câu hỏi gốc. Mục đích của đối chứng tráo là xem câu hỏi đúng có giúp chọn vùng hữu ích hơn không. Cả ba đầu vào ở bảng này dùng LR4 và Qwen gốc.

| Job và selector | Toàn ảnh | Crop query đúng | Crop query tráo | Δ query đúng − tráo | Token trung bình toàn / crop đúng / crop tráo |
| --- | ---: | ---: | ---: | ---: | --- |
| **11304 (CLIP chọn vùng)** | 65,51% | 42,95% | 33,17% | +9,78 [8,18; 11,43] | 930 / 782 / 782 |
| **11535 (SD-RPN chọn vùng)** | 65,75% | 56,52% | 46,12% | +10,39 [8,73; 12,07] | 930 / 453 / 572 |

Cùng 3.473 câu `val_inner`. SD-RPN tìm được ROI cho 2.996 câu với query đúng. Có 477 fallback ở query đúng và 1.545 ở query tráo. Chênh lệch giữa hai cách chọn box vì thế bao gồm cả tỷ lệ fallback và độ lớn box. Kiểm thêm trên 315 câu có crop ở cả hai nhánh và tỷ lệ token trong khoảng 0,8–1,25, query đúng đạt **65,17%**, query tráo **51,97%**, chênh **+13,21** [8,98; 17,57]. Phép thử crop ngẫu nhiên cùng kích thước ở job 12403 kiểm soát kích thước trực tiếp hơn.

SD-RPN dùng checkpoint Qwen2.5-VL-3B K21/T3 của Danh cho bước chọn vùng. `288` token tối đa và ngưỡng ROI `0.15` thuộc cấu hình selector ở các job SD-RPN. Qwen đọc ảnh và trả lời dùng checkpoint Qwen gốc của dự án, có thể gắn thêm module S1/A1/A2. Phần nguồn ở mục 8 ghi path của cả hai.

### Bật phục hồi trên toàn ảnh và crop gốc

**Job 11841** giữ Qwen và box SD-RPN, chỉ bật thêm S1b trên từng đầu vào. Trên cùng 3.473 câu, toàn ảnh tăng **65,58% → 67,47%**, tức +1,89 [0,95; 2,84]. Crop tăng **56,61% → 58,57%**, tức +1,96 [1,09; 2,85]. Trong nhóm 2.996 câu có crop thật, S1b tăng +1,77 điểm trên crop. Restorer được train trên toàn ảnh rồi áp trực tiếp lên lưới của crop.

Ở 2.996 crop thật của job này, bật S1b làm **95 câu đổi từ 0 lên 1** và **62 câu đổi từ 1 xuống 0**. Box và token budget giữ nguyên giữa hai nhánh. S1b thay vector đưa vào LLM nên có thể giúp đọc đúng ở một số câu và làm mất tín hiệu ở câu khác. Mức tăng trung bình tính cả những lần đổi điểm từng phần.

**Job 11920** dùng lại cùng kiểu crop và mở rộng sang bốn checkpoint. Cột crop toàn tập gồm 477 câu dùng fallback toàn ảnh. Hai cột cuối chỉ so trên đúng 2.996 câu có ROI. Tất cả đều là LR4.

| Nhánh | Toàn ảnh, 3.473 câu | Crop + fallback, 3.473 câu | Toàn ảnh, 2.996 câu có ROI | Crop thật, 2.996 câu |
| --- | ---: | ---: | ---: | ---: |
| **A0 (Qwen gốc)** | 65,70% | 56,43% | 69,16% | 58,40% |
| **S1a (khớp feature sạch)** | 64,82% | 55,45% | 68,22% | 57,36% |
| **S1b (khớp feature sạch + học đáp án)** | 67,41% | 58,70% | 70,71% | 60,62% |
| **A1 (visual adapter học đáp án)** | 67,73% | 58,94% | 70,94% | 60,76% |
| **A2 (LoRA ở LLM)** | **68,51%** | **59,65%** | **71,50%** | **61,23%** |

Trên 2.996 ROI, A2 trên crop kém A2 toàn ảnh **10,27 điểm** [8,86; 11,71], đổi lại giảm khoảng **59%** visual token, từ khoảng 933 xuống 380. Crop A2 hơn crop S1b **0,61 điểm** [−0,21; +1,46]. Con số **71,52%** là điểm tính lại từ dự đoán A2 toàn ảnh của job 11305 sau khi lọc đúng 2.996 câu có ROI. Điểm toàn tập của 11305 là **68,51%** trên 3.473 câu.

### EDA — crop làm sai ở đâu

Trên **2.996 crop thật** của job 11920, A2 chuyển **346 câu từ điểm 1 xuống 0** khi bỏ toàn ảnh và chuyển **80 câu từ 0 lên 1**. Trong 346 câu mất điểm hoàn toàn, 264 câu nhận điểm 0 ở cả năm nhánh crop. Điều này dẫn tới kiểm tra box và lượng thông tin được giữ.

| Phần vùng chữ đáp án nằm trong crop | Câu có grounding | A2 toàn ảnh | A2 crop | Chênh crop − toàn ảnh |
| --- | ---: | ---: | ---: | ---: |
| Dưới 10% | 69 | 59,86% | 20,29% | −39,57 |
| Từ 10% đến dưới 90% | 81 | 74,07% | 41,11% | −32,96 |
| Ít nhất 90% | 631 | 82,28% | 80,21% | −2,07 |

Bảng dùng 781 câu khớp duy nhất vùng TextOCR với đáp án đồng thuận ít nhất 8/10, phủ 687 ảnh. Đây là phân tích hậu nghiệm trên nhóm chữ dễ gán vùng. Với 150 câu giữ dưới 90% vùng đáp án, mở box 10% và 20% mỗi cạnh đưa lần lượt 41 và 54 câu lên mức phủ ít nhất 90%. Trong 69 câu bỏ lỡ gần trọn vùng đáp án, nới 20% cứu được 4 câu theo tiêu chí diện tích.

| Dạng lỗi quan sát | Ví dụ và ý nghĩa |
| --- | --- |
| **Box lệch khỏi chữ cần đọc** | Câu **13024** mất số `16` ngoài box. Nới 20% vẫn thiếu. Toàn ảnh giữ được thông tin. |
| **Box chạm chữ nhưng cắt mất một phần** | Câu **8913** cắt chữ `FRANC` sát mép. Nới box giữ được chữ. |
| **Hai câu cùng ảnh cần thông tin khác nhau** | Box chung ở **27450/27451** đủ để trả lời `2` ở câu đầu, nhưng câu sau cần `1/2 MILE`. Ở câu **14505**, mở rộng crop giúp A2 đổi từ `metro card reader` sang `meter`. |
| **Có vùng chữ nhưng trả lời sai nội dung hoặc định dạng** | Câu **6007** có lỗi cách ghi `bar bican`. Diện tích box chỉ giải thích một phần lỗi đọc. |
| **Crop đúng vùng có thể đủ hoặc có lợi** | Câu **18301** hỏi số lớn nhất trên thước, giữ được đáp án `71` với 180 token thay vì 999. Câu **7405** chỉ crop + S1b khôi phục được `250 kpl` trong các nhánh đã đối chiếu. |

Resize crop cũng thay đổi lưới Qwen. Có 175 crop dùng ít nhất bằng số token của toàn ảnh sau zoom. Có 103 crop chỉ còn tối đa 16 token, điểm crop 10,97% so với 52,82% của toàn ảnh trên nhóm này. Có 24 box trùng toàn ảnh về hình học nhưng bước zoom làm câu trả lời đổi ở 7 trường hợp. Các dấu hiệu này dẫn tới ablation zoom, padding và fallback.

## 6. Nới crop, bỏ zoom và ghép S1b với A2

### Job 12253 — giữ A2, thay cách chuẩn bị crop

Tất cả dùng **A2 (LoRA ở LLM)**, trên cùng 3.473 câu LR4. Mỗi cấu hình có số token tự nhiên sau khi xử lý ảnh. Padding 20% mỗi cạnh nghĩa là mở thêm 20% chiều rộng box ở cả trái và phải, 20% chiều cao ở cả trên và dưới, rồi cắt theo biên ảnh.

| Đầu vào A2 | Accuracy | Visual token trung bình | Δ so với crop gốc |
| --- | ---: | ---: | ---: |
| Toàn ảnh | **68,35%** | 930 | +8,75 |
| Crop SD-RPN gốc | 59,60% | 453 | — |
| Crop gốc, bỏ zoom riêng của crop | 60,14% | 290 | +0,54 |
| Crop nới 10% mỗi cạnh | 62,07% | 523 | +2,47 |
| Crop nới 20% mỗi cạnh | 63,14% | 582 | **+3,54** |
| Crop gốc + fallback theo token | 61,65% | 495 | +2,05 |

Padding 20% tăng **+3,54** [2,60; 4,48] trên đủ tập và **+4,10** [3,05; 5,19] trên nhóm có ROI. Bỏ zoom tiết kiệm token, với CI của chênh điểm [−0,37; +1,44]. Nhánh fallback dùng toàn ảnh cho 477 câu thiếu ROI, 174 câu crop có tối đa 25 token và 175 câu crop không tiết kiệm token. Tổng cộng 826 câu fallback và 2.647 câu dùng crop. Cấu hình pad20 vẫn thấp hơn toàn ảnh **5,21 điểm** [4,13; 6,32] trong job này.

### Job 12288 — cả hai loại module cùng lúc

Thử `S1b → A2` khi suy luận. S1b sửa visual feature rồi LLM có A2 đọc feature đó. Hai checkpoint đã được train riêng. Bảng giữ cùng 3.473 câu LR4. Toàn ảnh dùng trung bình 930 token, crop nới 20% cùng fallback dùng 582 token.

| Module | Toàn ảnh | Crop nới 20% + fallback |
| --- | ---: | ---: |
| **A0 (Qwen gốc)** | 65,59% | 59,65% |
| **S1b (restorer visual)** | 67,44% | 62,02% |
| **A2 (LoRA ở LLM)** | **68,53%** | **63,27%** |
| **S1b + A2 (restorer trước LLM có LoRA)** | 67,26% | 62,58% |

Ghép hai checkpoint làm điểm giảm so với A2 riêng **1,26** [0,66; 1,88] trên toàn ảnh và **0,68** [0,10; 1,27] trên crop. Kết quả áp dụng cho cách ghép tại inference. Huấn luyện chung hai module là thí nghiệm khác, chưa có kết quả.

## 7. Toàn ảnh + crop với đúng 576 visual token

### 12289 và 12376 — kiểm ảnh nào nên bật S1b

Qwen nhận hai vị trí ảnh trong một prompt, **toàn ảnh trước, crop SD-RPN nới 20% sau**. Ảnh giữ tỷ lệ hình và có đệm để tạo đúng lưới cần thiết. Đệm được tính vào ngân sách. Đối chứng toàn ảnh hoặc crop riêng nhận 576 visual token. Nhánh ghép nhận **288 + 288**, với hai token phân cách ảnh bổ sung trong prompt. Job 12376 chạy đủ tập và dùng toàn ảnh 576 token cho 477 câu thiếu box. Khi fallback, nhánh chỉ bật S1b ở crop dùng toàn ảnh gốc. Nhánh có bật S1b ở toàn ảnh dùng toàn ảnh đã phục hồi.

| Đầu vào | Module | Job 12289, 2.996 câu có ROI | Job 12376, đủ 3.473 câu |
| --- | --- | ---: | ---: |
| Toàn ảnh | A0 (Qwen gốc) | 67,10% | 63,85% |
| Toàn ảnh | S1b (restorer visual) | 69,27% | 66,25% |
| Toàn ảnh | A2 (LoRA ở LLM) | — | **67,15%** |
| Crop | A0 (Qwen gốc) | 60,31% | 57,87% |
| Crop | S1b (restorer visual) | 62,24% | 60,21% |
| Toàn ảnh + crop | Qwen gốc, cả hai ảnh giữ nguyên feature | 68,57% | 65,01% |
| Toàn ảnh + crop | S1b chỉ ở toàn ảnh | 69,77% | 66,75% |
| Toàn ảnh + crop | S1b chỉ ở crop | 69,81% | 66,06% |
| Toàn ảnh + crop | S1b ở cả hai | **69,98%** | 66,87% |

- Với Qwen gốc và cùng 576 token, chia token cho toàn ảnh + crop tăng **1,47** [0,29; 2,64] điểm ở nhóm có ROI. Trên đủ 3.473 câu, mức tăng **1,16** [0,11; 2,20] vẫn dương.
- Trên đủ tập, ghép cả hai ảnh đều phục hồi đạt 66,87%. So với S1b toàn ảnh, chênh **+0,62** [−0,37; +1,61]. So với A2 toàn ảnh, chênh **−0,28** [−1,36; +0,80]. Cấu hình ghép Qwen gốc kém A2 toàn ảnh **2,14 điểm** [0,94; 3,33]. Cả hai phép so A2 dùng mốc **67,15% trong cùng job**, cùng số câu và token.

### 12403 — giữ A2 ở cả ba nhánh trên validation chính thức

Job chạy trên **5.000 câu LR4, 3.166 ảnh**. Cả ba nhánh giữ cùng checkpoint A2 và cùng ngân sách **576 token**. Đối chiếu 15.000 dòng dự đoán xác nhận ba nhánh có cùng cặp image ID và question ID, mỗi dòng dùng đúng 576 visual token. Nhánh crop ngẫu nhiên giữ kích thước box của SD-RPN và thử đặt lại vị trí với seed 42. Hai nhánh ghép cùng fallback toàn ảnh trên 746 câu thiếu ROI. Có 112 câu có ROI mà box ngẫu nhiên giữ vị trí cũ.

| Nhánh | Visual token | Accuracy | Độ trễ trung bình |
| --- | --- | ---: | ---: |
| **A2 toàn ảnh** | 576 | 65,16% | 220 ms/câu |
| **A2 toàn ảnh + crop SD-RPN** | 288 + 288, hoặc 576 khi fallback | **65,42%** | 367 ms/câu |
| **A2 toàn ảnh + crop ngẫu nhiên cùng kích thước** | 288 + 288, hoặc 576 khi fallback | 63,76% | 368 ms/câu |

| So sánh theo cặp trên đủ 5.000 câu | Δ điểm | CI 95% |
| --- | ---: | --- |
| Crop SD-RPN − toàn ảnh | **+0,26** | [−0,56; +1,07] |
| Crop SD-RPN − crop ngẫu nhiên | **+1,65** | [+0,97; +2,33] |
| Crop ngẫu nhiên − toàn ảnh | **−1,40** | [−2,28; −0,54] |

Hai nhánh ghép tốn thêm khoảng **147 ms/câu** so với A2 toàn ảnh, chủ yếu ở selector. Trên 4.254 câu có ROI, ghép SD-RPN hơn toàn ảnh **+0,30** [−0,64; +1,28]. Đối chứng cùng kích thước cho thấy chọn đúng vị trí có ích. Mức chênh với toàn ảnh vẫn chưa đủ rõ để đổi mốc A2 toàn ảnh ở ngân sách 576.

## 8. Nguồn đối chiếu và những phần còn mở

### Checkpoint và artifact

**Qwen đọc và trả lời** dùng `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. **SD-RPN của Danh** dùng cho chọn box trong các job crop từ 11535 trở đi. Checkpoint selector trên HPC nằm tại đường dẫn dưới đây.

```text
/datastore/dungmt/textvqa/Danh/tmp/huggingface/hub/models--YuhengSSS--qwen2_5vl-3b-roi-K21T3-152k-v1bf16Mheads-twiginit-filled/snapshots/f6b9e7e188d7524f3df871003b00e9ba6102d929
```

**Checkpoint S1a, S1b, A1, A2, A1clean, A2clean** từ 11305 nằm dưới thư mục này. Mỗi nhánh có file `checkpoint.pt` trong thư mục mang tên nhánh.

```text
/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA/runs/s1-20261107T051836Z-wymhdtwk/s1/variants/
```

Hai gốc lưu run trên HPC được ký hiệu MAIN và CROP.

```text
MAIN  /datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA/runs/
CROP  /datastore/dungmt/textvqa/Tuan/Q-LRG-crop/runs/
```

| Job | Gốc | Run directory | File đối chiếu |
| --- | --- | --- | --- |
| 10684 | MAIN | `smoke12-20261106T045704Z-8zvdfo28` | `d0/identity.json`, `s0/decision.md` |
| 10756 | MAIN | `full500-20261106T120210Z-6z0yzfi_` | `d0/baseline_summary.csv`, `s0/oracle_summary.csv`, `s0/oracle_coverage.json` |
| 10820 | MAIN | `s1-20261106T161413Z-lm0jjs3z` | `s1/summary.csv`, `s1/baseline_audit.json`, `s1/review.html` |
| 11172 | MAIN | `audit-20261107T042743Z-3zuwy0dc` | `audit/summary.json`, `audit/pilot_split_audit.json`, `datasets/` |
| 11188 / 11305 | MAIN | `s1-20261107T051836Z-wymhdtwk` | `s1/summary.csv`, `s1/degradation_interaction.csv`, `s1/training_history.csv`, `s1/review.html`, `s1/review-clean.html` |
| 11304 | CROP | `query-crop-20261107T070738Z-k7u86oxx` | `summary.json`, `predictions.jsonl` |
| 11535 | CROP | `sdrpn-crop-20261107T173143Z-heum7_b7` | `summary.json`, `predictions.jsonl`, `review.html` |
| 11841 | CROP | `sdrpn-restore-20261108T043912Z-sruu5jmr` | `summary.json`, `predictions.jsonl`, `review.html` |
| 11920 | CROP | `sdrpn-restore-20261108T115242Z-ea3mmyhy` | `summary.json`, `predictions.jsonl`, `review.html` |
| 12253 | CROP | `sdrpn-ablation-20261109T110200Z-qf13mfvv` | `summary.json`, `predictions.jsonl`, `review.html` |
| 12288 | CROP | `sdrpn-stack-20261109T150949Z-lx5fkb8m` | `summary.json`, `predictions.jsonl`, `review.html` |
| 12289 | CROP | `sdrpn-fusion-20261109T152551Z-glgd60j4` | `summary.json`, `predictions.jsonl`, `review.html` |
| 12376 | CROP | `sdrpn-fusion-all-20261110T085508Z-8ra6byif` | `summary.json`, `predictions.jsonl`, `review.html` |
| 12403 | CROP | `a2-fusion-official-20261110T115853Z-k5mhzwkw` | `summary.json`, `predictions.jsonl`, `review.html`, `run.json` |

`summary` dùng để tra điểm và so sánh theo cặp. `predictions.jsonl` tra từng câu, đối chiếu đáp án sinh ra và số token. `review.html` lọc nhánh, mã câu và trường hợp điểm tăng hoặc giảm. Các kết quả chính trong tài liệu dựa trên artifact của từng job và trạng thái Slurm. Bảng EDA dùng phép ghép theo câu và vùng OCR trên artifact 11920, nên chỉ áp dụng cho cohort ghi ngay dưới bảng.

### Phần đã có bằng chứng và phần cần thêm thí nghiệm

| Hạng mục | Trạng thái tại mốc 12403 |
| --- | --- |
| **Restorer toàn ảnh** | S1b tăng so với Qwen gốc. A2 là đối chứng có điểm LR4 cao nhất trong 11305. Cần tách tác dụng phục hồi khỏi tác dụng học trả lời nếu tiếp tục theo RQ1. |
| **Restoration theo vùng chữ** | S0 đã chỉ ra lợi ích của feature sạch vùng chữ. Các lượt S1 đã chạy dùng loss feature toàn ảnh. Đo cosine riêng ở token chữ và ngoài chữ, thêm loss có trọng số TextOCR chỉ lúc train, so mask center với overlap và đối chứng ngẫu nhiên cỡ O2 là các phép thử còn mở. |
| **Chọn vùng theo câu hỏi** | SD-RPN đã được thử như một bộ chọn box ở đầu vào. Chọn box có lợi so với đối chứng ngẫu nhiên cùng kích thước. Gate phục hồi S2/S3 trên lưới latent vẫn ở phần thiết kế. |
| **Phân bổ token** | Đã thử chia 288 + 288. Các tỷ lệ khác hoặc chính sách chọn giữ bao nhiêu token cho toàn ảnh theo từng câu vẫn còn mở. |
| **Phạm vi xác nhận** | Kết quả CLEAN/LR4 của đủ sáu checkpoint và A0 hiện ở `val_inner`, seed 42. `official_val` đã có job xác nhận ba đầu vào LR4 với A2. Đa seed, các suy giảm khác và test chính thức chưa có kết quả trong chuỗi này. |
