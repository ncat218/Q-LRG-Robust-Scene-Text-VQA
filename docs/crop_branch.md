# Nhánh X — Phục hồi latent trên crop

Phần của Danh chọn token ở vùng chữ để đưa vào LLM và nghiên cứu giảm chi phí tính toán. Phần này đưa **ảnh toàn cảnh hoặc ảnh crop theo câu hỏi** qua vision encoder của Qwen rồi phục hồi feature sau Merger. Phép so chính đo lợi ích của phục hồi và lượng ngữ cảnh cần giữ. Các mã C0–C4 trong tài liệu thuộc nhánh crop.

Checkout riêng trên `crop-branch`. Tái sử dụng manifest, Qwen wrapper, scorer MMF và bootstrap theo ảnh. C2 có thể dùng cơ chế injection sẵn có để thử restorer S1 trên crop. Nhánh crop có thư mục run riêng và dùng môi trường Q-LRG đã cài để chạy inference.

| Arm | Đầu vào | Trạng thái |
| --- | --- | --- |
| C0 | LR4 toàn cảnh, Qwen gốc | Có trong pilot |
| C1 | Crop oracle theo vùng đáp án TextOCR trên LR4, zoom Bicubic, Qwen gốc | Có trong pilot |
| Random | Crop LR4 ngẫu nhiên cùng chiều rộng, chiều cao và mức zoom của C1 | Có trong pilot |
| C2 | Cùng crop C1, thêm phục hồi latent | Chờ kết quả C1. Có thể thử checkpoint S1 trước để đo khả năng chuyển sang crop. Huấn luyện trên crop là lượt riêng. |
| C3 | Toàn cảnh và cùng crop đã chọn | Chờ kết quả C1. So với crop riêng để đo lợi ích ngữ cảnh, đồng thời ghi số token tăng thêm. |
| C4 | Crop do detector và bộ xếp hạng theo câu hỏi chọn | Chờ kết quả C1. So với crop oracle để đo mức mất do chọn vùng. |

Pilot giữ cùng câu hỏi, Qwen và cách chấm ở C0/C1/Random. C1 cắt từ ảnh LR4 đã suy giảm, thêm padding mặc định 10% theo từng chiều rồi zoom Bicubic 2×. Đây là lựa chọn khởi đầu. Random dùng seed 42 và lấy vị trí khác nếu có thể. Báo cáo giữ tỷ lệ vùng đáp án được crop bao phủ vì một số crop ngẫu nhiên vẫn có thể chạm vùng chữ.

Lượt đầu tái sử dụng cohort grounding tạm của S0 job 10756. Có thể cung cấp JSONL đã duyệt bằng `--review` để chạy trên cohort mới của cùng manifest. C1 dùng nhãn vùng đáp án nên chỉ đo tiềm năng chọn vùng. Cần thấy lợi ích so với C0 và xem khoảng tin cậy ghép cặp trước khi cấp ngân sách cho C2–C4. So với Random để đánh giá vai trò của vị trí chọn vùng. Cohort grounding tạm ưu tiên đáp án khớp OCR đơn nhất, nên khi báo cáo cần ghi phạm vi này và số token sau resize.

## Chạy trên UIT-HPC

Tạo worktree riêng từ nhánh trên GitHub. Dùng thư mục chứa môi trường Q-LRG đang hoạt động nếu khác `.venv`.

```bash
main=/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA
crop=/datastore/dungmt/textvqa/Tuan/Q-LRG-crop
git -C "$main" fetch origin crop-branch
git -C "$main" worktree add --detach "$crop" FETCH_HEAD
cd "$crop"

python3 run_crop.py --env "$main/.venv" \
  --from-run "$main/runs/full500-20261106T120210Z-6z0yzfi_"
```

Launcher tạo một job MIG 35 GB với mặc định 30 phút. Thêm `--dry-run` để xem lệnh Slurm hoặc `--limit 12` để thử trên 12 câu. Job kiểm tra môi trường có sẵn trước khi load model. Kết quả ở thư mục `runs/crop-oracle-…` trong worktree crop, gồm `summary.json`, `predictions.jsonl` và `review.html` để lọc câu tăng, giảm hoặc giữ điểm. Mỗi dự đoán có tọa độ crop, kích thước trước processor, grid, số visual token và thời gian sinh.

Để đo recall detector trên LR4, thêm `--detections /absolute/path/lr4_detections.jsonl`. Mỗi ảnh trong cohort cần một dòng có `image_id`, `image_sha256` từ manifest, `condition: "LR4"` và `boxes` theo tọa độ `[x0,y0,x1,y1]` trên canvas LR4. Ghi danh sách rỗng khi detector bỏ sót. Báo cáo đếm mỗi từ vùng đáp án có ít nhất 50% diện tích bbox nằm trong một bbox dự đoán và đếm số câu giữ được mọi từ vùng đáp án. Kết quả detector dùng để tính recall trong lượt này.
