# Nhánh X — Phục hồi latent trên crop

Phần của Danh chọn token ở vùng chữ để đưa vào LLM và nghiên cứu giảm chi phí tính toán. Phần này đưa **ảnh toàn cảnh hoặc ảnh crop theo câu hỏi** qua vision encoder của Qwen rồi phục hồi feature sau Merger. Phép so chính đo lợi ích của phục hồi và lượng ngữ cảnh cần giữ. Các mã C0–C4 trong tài liệu thuộc nhánh crop.

Checkout riêng trên `crop-branch`. Tái sử dụng manifest, Qwen wrapper, scorer MMF và bootstrap theo ảnh. Nhánh crop có thư mục run riêng và dùng môi trường Q-LRG đã cài để chạy inference.

## Crop theo câu hỏi

Mặc định dùng SD-RPN Qwen2.5-VL-3B K21/T3 để đề xuất vùng từ ảnh LR4 và câu hỏi, với tối đa 288 token ở bước chọn vùng và ngưỡng 0,15. Bước này dùng môi trường Transformers 4.51.3 riêng. Trong cùng job, môi trường Q-LRG nạp Qwen gốc sau khi tiến trình chọn vùng thoát. Nó kiểm tra hash pixel LR4 của hai bước, cắt hộp SD-RPN và zoom bicubic 1,5×. Nếu SD-RPN không tìm thấy vùng, Qwen nhận toàn ảnh LR4 và báo cáo ghi ca fallback. Các đầu vào của bộ chọn chỉ chứa ảnh và văn bản câu hỏi. Đáp án chuẩn được dùng ở bước chấm sau dự đoán. [Paper SD-RPN](https://openreview.net/pdf/592acc72239cdc9214dbc4b0e1b63dcc8d2e311f.pdf)

Job 11304 dùng baseline CLIP ViT-B/32 với tối đa 16 cửa sổ cố định, kích thước 60% chiều rộng và chiều cao ảnh. Dùng `--selector clip` để tái lập cách chọn này. SD-RPN và CLIP có thể chọn vùng khác kích thước, nên so sánh giữa chúng cần báo cáo cả số visual token và tỷ lệ fallback.

| Nhánh | Đầu vào |
| --- | --- |
| `full` | Toàn ảnh LR4, giữ nguyên visual feature của Qwen |
| `query` | Crop theo query, giữ nguyên visual feature |
| `shuffled` | Chọn vùng bằng câu hỏi khác từ một ảnh khác. Qwen vẫn trả lời câu hỏi gốc. |
| `full_restore` | Toàn ảnh LR4, sửa feature sau Merger bằng checkpoint S1 |
| `query_restore` | Đúng ảnh crop của `query`, sửa feature bằng cùng checkpoint S1 |

`query` so với `shuffled` đo tác động của query lên chọn vùng. Ở baseline CLIP kích thước crop cố định, còn SD-RPN có thể đổi cả vị trí, kích thước và quyết định fallback. Khi hai nhánh có cùng ảnh đầu vào và câu hỏi, dùng lại dự đoán. `query` so với `full` đo tác động tổng của crop và zoom. `full_restore` so với `full` và `query_restore` so với `query` đo riêng tác động của phục hồi trên hai dạng đầu vào. `query_restore` so với `full_restore` đo hiệu quả khi chuyển cùng restorer từ toàn ảnh sang crop. Các nhánh đi qua cùng đường injection, với feature gốc làm mốc.

Chạy trên `val_inner` từ manifest audit. Mặc định dùng toàn bộ split. Dùng `--images 500` nếu cần một tập con cố định theo seed. Khoá cấu hình và checkpoint trên dev trước khi dùng `--split official_val`. Phép so toàn ảnh + crop là lượt riêng.

```bash
main=/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA
crop=/datastore/dungmt/textvqa/Tuan/Q-LRG-crop
data="$main/runs/audit-20261107T042743Z-3zuwy0dc/datasets"

cd "$crop"
git pull --ff-only origin crop-branch
python3 run_query_crop.py --env "$main/.venv" --data-dir "$data" \
  --sdrpn-env /path/to/sdrpn-venv --sdrpn-vendor /path/to/vendor \
  --sdrpn-checkpoint /path/to/sdrpn-3b-snapshot
```

Khi job S1 đã có `s1/COMPLETE.json`, thêm `--s1-run` và `--arm` để chạy đủ năm nhánh. Ví dụ sau chọn S1b. Chốt arm trên dev trước khi đánh giá cuối.

```bash
python3 run_query_crop.py --env "$main/.venv" --data-dir "$data" \
  --sdrpn-env /path/to/sdrpn-venv --sdrpn-vendor /path/to/vendor \
  --sdrpn-checkpoint /path/to/sdrpn-3b-snapshot \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" --arm S1b
```

Để thử phục hồi trên đúng crop đã chấm ở `val_inner`, dùng `--from-crop-run`. Launcher thừa hưởng cohort, zoom và môi trường Qwen của run nguồn. Job kiểm tra checksum dự đoán nguồn, dùng lại tọa độ và chạy bốn nhánh `full`, `full_restore`, `query`, `query_restore`. Phục hồi sửa feature sau Merger và giữ nguyên kích thước grid cùng toàn bộ visual token. Chế độ này chấp nhận một nhánh S1 đã lưu checkpoint cuối và `training.json` khi các nhánh S1 khác còn chạy. Worker xác minh số mẫu, số bước, lịch sử huấn luyện và binding trước khi nạp restorer.

```bash
python3 run_query_crop.py \
  --from-crop-run "$crop/runs/sdrpn-crop-20261107T173143Z-heum7_b7" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" --arm S1b
```

Dùng `--arms S1a S1b A1` thay cho `--arm S1b` để so các checkpoint trong một job replay.
S1a học khớp feature sạch, S1b học thêm từ đáp án và A1 là visual adapter chỉ học từ đáp án.
Mỗi đầu vào chạy Qwen gốc và ba checkpoint trên cùng feature đầu vào. Cả ba giữ đủ token của crop.
`summary.json` ghi SHA-256 từng checkpoint và CI ghép cặp giữa các checkpoint trên toàn ảnh, crop, và nhóm có crop thật.

Kết quả nằm trong `runs/sdrpn-restore-…`. `summary.json` báo cáo cả tập và nhóm `roi_only` có crop thật, tách các ca quay về toàn ảnh. `review.html` tham chiếu ảnh xem trước đã lưu ở run SD-RPN nguồn. Giữ các đường dẫn tương đối của hai run khi tải cả HTML và ảnh về.

Launcher submit ngay. SD-RPN ghi tọa độ và kết quả trong `runs/sdrpn-crop-…`, với một file đề xuất mỗi ảnh để hỗ trợ resume. Checkpoint và môi trường SD-RPN có sẵn được đọc tại chỗ. Lượt `--selector clip` ghi trong `runs/query-crop-…` và lấy checkpoint CLIP ghim revision khi cache trống (khoảng 605 MB). Có thể chuẩn bị cache CLIP tại node có mạng từ thư mục crop bằng lệnh:

```bash
"$main/.venv/bin/python" -B -c 'from scripts.query_crop import selector_snapshot; print(selector_snapshot("local/selector-cache"))'
```

Launcher kiểm tra import `pyexpat` và phiên bản Transformers của SD-RPN trước khi submit. Nếu Python Conda báo thiếu symbol Expat, thêm `--sdrpn-libexpat /path/to/miniconda3/lib/libexpat.so.1`. Thư viện này chỉ được nạp trong tiến trình SD-RPN.

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
