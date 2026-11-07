# Nhánh X — Phục hồi latent trên crop

Phần của Danh chọn token ở vùng chữ để đưa vào LLM và nghiên cứu giảm chi phí tính toán. Phần này đưa **ảnh toàn cảnh hoặc ảnh crop theo câu hỏi** qua vision encoder của Qwen rồi phục hồi feature sau Merger. Phép so chính đo lợi ích của phục hồi và lượng ngữ cảnh cần giữ. Các mã C0–C4 trong tài liệu thuộc nhánh crop.

Checkout riêng trên `crop-branch`. Tái sử dụng manifest, Qwen wrapper, scorer MMF và bootstrap theo ảnh. Nhánh crop có thư mục run riêng và dùng môi trường Q-LRG đã cài để chạy inference.

## Crop theo câu hỏi

Tạo tối đa 16 cửa sổ chồng lấn từ ảnh LR4. Mỗi cửa sổ mặc định có chiều rộng và chiều cao bằng 60% canvas. [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268) đóng băng chọn vùng có cosine lớn nhất giữa embedding của cửa sổ và embedding câu hỏi. Crop được zoom bicubic 1,5× trước khi đưa vào Qwen. Bộ chọn chỉ dùng pixel LR4 và văn bản câu hỏi. Đáp án chuẩn được dùng sau khi sinh dự đoán để tính điểm.

| Nhánh | Đầu vào |
| --- | --- |
| `full` | Toàn ảnh LR4, giữ nguyên visual feature của Qwen |
| `query` | Crop theo query, giữ nguyên visual feature |
| `shuffled` | Cùng tập cửa sổ, chọn bằng query khác từ một ảnh khác. Qwen vẫn trả lời query gốc. |
| `full_restore` | Toàn ảnh LR4, sửa feature sau Merger bằng checkpoint S1 |
| `query_restore` | Đúng ảnh crop của `query`, sửa feature bằng cùng checkpoint S1 |

`query` so với `shuffled` đo tác động của query lên chọn vùng với cùng kích thước crop. Khi hai query chọn cùng vùng thì dùng lại dự đoán vì Qwen nhận cùng ảnh và câu hỏi. `query` so với `full` đo tác động tổng của crop và zoom. `full_restore` so với `full` và `query_restore` so với `query` đo riêng tác động của phục hồi trên hai dạng đầu vào. `query_restore` so với `full_restore` đo hiệu quả khi chuyển cùng restorer từ toàn ảnh sang crop. Nếu crop kém, phép so này chưa phân biệt được lỗi chọn vùng, mất ngữ cảnh và mức độ phù hợp của checkpoint huấn luyện trên toàn ảnh. Các nhánh đi qua cùng đường injection, với feature gốc làm mốc.

Chạy trên `val_inner` từ manifest audit. Mặc định dùng toàn bộ split. Dùng `--images 500` nếu cần một tập con cố định theo seed. Khoá cấu hình và checkpoint trên dev trước khi dùng `--split official_val`. Baseline này dùng cửa sổ ảnh. Detector của nhóm và phép so toàn ảnh + crop là các thử nghiệm riêng.

```bash
main=/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA
crop=/datastore/dungmt/textvqa/Tuan/Q-LRG-crop
data="$main/runs/audit-20261107T042743Z-3zuwy0dc/datasets"

# Trong checkout crop sau khi job oracle đã dừng hoặc hoàn tất.
cd "$crop"
git pull --ff-only origin crop-branch
python3 run_query_crop.py --env "$main/.venv" --data-dir "$data"
```

Khi job S1 đã có `s1/COMPLETE.json`, thêm `--s1-run` và `--arm` để chạy đủ năm nhánh. Ví dụ sau chọn S1b. Chốt arm trên dev trước khi đánh giá cuối.

```bash
python3 run_query_crop.py --env "$main/.venv" --data-dir "$data" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" --arm S1b
```

Launcher submit ngay. Slurm worker dùng môi trường có sẵn, lấy checkpoint CLIP ghim revision khi cache trống (khoảng 605 MB) và ghi kết quả riêng trong `runs/query-crop-…`. Cache mặc định ở `local/selector-cache` của checkout crop. Nếu compute node không truy cập được Hugging Face, có thể chuẩn bị cache tại node có mạng từ thư mục crop bằng lệnh:

```bash
"$main/.venv/bin/python" -B -c 'from scripts.query_crop import selector_snapshot; print(selector_snapshot("local/selector-cache"))'
```

Khi cần tiếp tục một lượt bị ngắt, chạy `python3 run_query_crop.py --resume /absolute/path/to/run --time 06:00:00` từ checkout ban đầu. Worker tái sử dụng các shard đã ghi xong theo ảnh.

Kết quả gồm `summary.json` (điểm, CI ghép cặp theo ảnh, số token, thời gian và số vùng trùng nhau), `predictions.jsonl` (tọa độ, điểm và câu trả lời), `review.html` (lọc tăng/giảm và tìm theo mã câu) cùng thư mục ảnh `review-images/`. Giữ HTML và thư mục ảnh cạnh nhau khi tải về.

```bash
tar -czf "${run%/}-review.tar.gz" -C "$run" \
  run.json COMPLETE.json summary.json predictions.jsonl review.html review-images
```

## Crop oracle để chẩn đoán

Launcher `run_crop.py` chạy oracle của job 11223. Có thể tham khảo để ước lượng tiềm năng khi đã biết vùng đáp án.

| Arm | Đầu vào | Trạng thái |
| --- | --- | --- |
| C0 | LR4 toàn cảnh, Qwen gốc | Có trong pilot |
| C1 | Crop oracle theo vùng đáp án TextOCR trên LR4, zoom Bicubic, Qwen gốc | Có trong pilot |
| Random | Crop LR4 ngẫu nhiên cùng chiều rộng, chiều cao và mức zoom của C1 | Có trong pilot |

Pilot giữ cùng câu hỏi, Qwen và cách chấm ở C0/C1/Random. C1 cắt từ ảnh LR4 đã suy giảm, thêm padding mặc định 10% theo từng chiều rồi zoom Bicubic 2×. Đây là lựa chọn khởi đầu. Random dùng seed 42 và lấy vị trí khác nếu có thể. Báo cáo giữ tỷ lệ vùng đáp án được crop bao phủ vì một số crop ngẫu nhiên vẫn có thể chạm vùng chữ.

Lượt đầu tái sử dụng cohort grounding tạm của S0 job 10756. Có thể cung cấp JSONL đã duyệt bằng `--review` để chạy trên cohort mới của cùng manifest. C1 dùng nhãn vùng đáp án nên chỉ đo tiềm năng khi vùng đã biết. So với Random để đánh giá vai trò của vị trí chọn vùng. Cohort grounding tạm ưu tiên đáp án khớp OCR đơn nhất, nên khi báo cáo cần ghi phạm vi này và số token sau resize.

Launcher tạo một job MIG 35 GB với mặc định 30 phút. Thêm `--dry-run` để xem lệnh Slurm hoặc `--limit 12` để thử trên 12 câu. Job kiểm tra môi trường có sẵn trước khi load model. Kết quả ở thư mục `runs/crop-oracle-…` trong worktree crop, gồm `summary.json`, `predictions.jsonl` và `review.html` để lọc câu tăng, giảm hoặc giữ điểm. Mỗi dự đoán có tọa độ crop, kích thước trước processor, grid, số visual token và thời gian sinh.

Để đo recall detector trên LR4, thêm `--detections /absolute/path/lr4_detections.jsonl`. Mỗi ảnh trong cohort cần một dòng có `image_id`, `image_sha256` từ manifest, `condition: "LR4"` và `boxes` theo tọa độ `[x0,y0,x1,y1]` trên canvas LR4. Ghi danh sách rỗng khi detector bỏ sót. Báo cáo đếm mỗi từ vùng đáp án có ít nhất 50% diện tích bbox nằm trong một bbox dự đoán và đếm số câu giữ được mọi từ vùng đáp án. Kết quả detector dùng để tính recall trong lượt này.
