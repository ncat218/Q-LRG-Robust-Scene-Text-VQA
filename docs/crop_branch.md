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

Dùng `--arms S1a S1b A1 A2` thay cho `--arm S1b` để so các checkpoint trong một job replay.
S1a học khớp feature sạch, S1b học thêm từ đáp án và A1 là visual adapter chỉ học từ đáp án.
A2 gắn LoRA vào Q/V của attention trong LLM khi sinh đáp án và dùng nguyên feature đầu vào.
Mỗi đầu vào chạy Qwen gốc và bốn checkpoint trên cùng feature đầu vào. Tất cả giữ đủ token của crop.
`summary.json` ghi SHA-256 từng checkpoint và CI ghép cặp giữa các checkpoint trên toàn ảnh, crop, và nhóm có crop thật.

Kết quả nằm trong `runs/sdrpn-restore-…`. `summary.json` báo cáo cả tập và nhóm `roi_only` có crop thật, tách các ca quay về toàn ảnh. `review.html` tham chiếu ảnh xem trước đã lưu ở run SD-RPN nguồn. Giữ các đường dẫn tương đối của hai run khi tải cả HTML và ảnh về.

### Kiểm tra zoom, padding và fallback

```bash
python3 run_query_crop.py \
  --from-crop-run "$crop/runs/sdrpn-crop-20261107T173143Z-heum7_b7" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" \
  --arms A2 --crop-ablation
```

Một job so A2 trên toàn ảnh, crop gốc, cùng crop bỏ phóng thêm, crop nới 10% và 20% mỗi phía, và luật fallback.
Box nới lấy pixel từ ảnh LR4, giới hạn ở biên ảnh và giữ mức zoom gốc. Nhánh bỏ phóng thêm vẫn dùng kích thước
tối thiểu và image processor như các nhánh còn lại. Luật fallback chọn toàn ảnh khi thiếu ROI, crop còn tối đa
25 token hoặc crop dùng ít nhất bằng số token của toàn ảnh. Nhánh này dùng dự đoán trong cùng job.
Giữ toàn bộ token đầu vào ở mọi nhánh. Đọc điểm, CI, số token và lý do fallback trong `summary.json` của
`runs/sdrpn-ablation-…`. `review.html` dùng cả ảnh xem trước mới và ảnh của run nguồn.

### Ghép S1b và LoRA A2

```bash
python3 run_query_crop.py \
  --from-crop-run "$crop/runs/sdrpn-crop-20261107T173143Z-heum7_b7" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" \
  --arms S1b A2 --stack-s1b-a2
```

Job dùng checkpoint đã train của S1b và A2, rồi chạy Qwen gốc, S1b, A2 và S1b + A2 trên từng đầu vào.
Hai đầu vào là toàn ảnh LR4 và crop SD-RPN nới 20% mỗi phía với mức zoom gốc, như nhánh pad20.
Nhánh ghép dùng đúng feature đã sửa của S1b và chỉ gắn LoRA A2 trong lượt sinh đáp án.
Đọc `summary.json` trong `runs/sdrpn-stack-…` để so bản ghép với A2 và S1b, trên cả tập và nhóm có crop thật.

### Toàn ảnh và crop với cùng ngân sách visual token

```bash
python3 run_query_crop.py \
  --from-crop-run "$crop/runs/sdrpn-crop-20261107T173143Z-heum7_b7" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" \
  --arms S1b --fair-fusion
```

Chấm cùng các câu có crop thật. Mọi nhánh dùng đúng 576 visual token. Toàn ảnh riêng và crop riêng mỗi bên
có hai mức tắt/bật S1b. Nhánh hai ảnh có 288 token toàn ảnh và 288 token crop, lần lượt thử tắt cả hai,
chỉ phục hồi toàn ảnh, chỉ phục hồi crop và phục hồi cả hai. Crop dùng box SD-RPN nới 20% mỗi phía.
Resize giữ tỷ lệ khung hình và thêm viền trung tính khi cần. Viền tính vào ngân sách và tỷ lệ vùng ảnh thật
được báo cáo riêng. Qwen gốc dùng thứ tự toàn ảnh rồi crop, mỗi ảnh có grid riêng. Số token toàn prompt
được báo cáo riêng vì nhánh hai ảnh có thêm dấu phân cách. Worker kiểm tra parity logit hai ảnh trước khi chấm.
Kết quả nằm trong `runs/sdrpn-fusion-…`.

Để chấm cùng tập 3.473 câu của S1, dùng `--arms S1b A2 --fair-fusion-all` thay cho hai cờ trên.
Ở các câu thiếu ROI, nhánh crop và fusion dùng toàn ảnh với 576 visual token. S1b chỉ chạy trên
toàn ảnh nếu cấu hình yêu cầu phục hồi toàn ảnh hoặc phục hồi toàn bộ đầu vào. Nhánh chỉ phục hồi crop
giữ nguyên feature khi phải quay về toàn ảnh. A2 toàn ảnh được chấm cùng 576 token.
Bảng chính gồm toàn bộ câu, còn `roi_only` giữ phép so phụ trên các câu có crop.

### Kiểm chứng A2 trên validation chính thức

```bash
python3 run_query_crop.py \
  --from-crop-run "$crop/runs/sdrpn-crop-20261107T173143Z-heum7_b7" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" \
  --arms A2 --confirm-a2-fusion
```

Chế độ này lấy cấu hình selector đã khóa từ run nguồn và tạo box mới cho 5.000 câu validation chính thức.
Ba nhánh giữ cùng A2 và 576 visual token. So toàn ảnh, toàn ảnh + crop SD-RPN nới 20%,
và toàn ảnh + crop ngẫu nhiên cùng kích thước. Nhánh fusion dùng 288 + 288 token và quay về toàn ảnh
khi thiếu ROI. Báo cáo có CI theo ảnh, nhóm phụ có crop, nhóm đổi được vị trí crop ngẫu nhiên và
thời gian selector cộng reader khi model đã được nạp. Kết quả ở `runs/a2-fusion-official-…`.

### Benchmark SD-RPN và DBNet trên toàn bộ val_inner

```bash
python3 run_query_crop.py \
  --from-crop-run "$crop/runs/sdrpn-crop-20261107T173143Z-heum7_b7" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" \
  --arms A2 --dbnet-benchmark \
  --dbnet-model "$crop/local/dbnet/DB_IC15_resnet18.onnx" \
  --dbnet-runtime "$crop/local/dbnet/python"
```

Đặt checkpoint DBNet ResNet-18 ICDAR2015 công khai vào đường dẫn `--dbnet-model`.
Worker kiểm tra SHA-256 `62fce3a058ac519952062fbd2cf45bae661130ce753d2a0fc918f4776ba3a02a`.
Dùng một Python có pip để cài `opencv-python-headless==4.10.0.84` với
`--no-deps --target "$crop/local/dbnet/python"`. DBNet chạy bằng CPU trong môi trường Q-LRG và dùng
chính các pixel LR4 của SD-RPN và A2. [Checkpoint và cấu hình mẫu của OpenCV](https://github.com/opencv/opencv/blob/4.12.0/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown)

| Nhánh | Đầu vào A2 |
| --- | --- |
| `full_A2` | Toàn ảnh |
| `sdrpn` | Toàn ảnh + box SD-RPN theo câu hỏi |
| `dbnet` | Toàn ảnh + box DBNet có confidence cao nhất |
| `hybrid_union` | Toàn ảnh + hình chữ nhật bao cả box SD-RPN và box DBNet đã chọn |
| `hybrid_text` | Toàn ảnh + nguyên box DBNet đã chọn |
| `hybrid_intersection` | Toàn ảnh + phần hai box giao nhau |
| `hybrid_random` | Toàn ảnh + box đặt ngẫu nhiên, cùng kích thước crop của `hybrid_union` sau padding |
| `hybrid_shuffled` | Toàn ảnh + box bao từ SD-RPN nhận câu hỏi tráo, A2 trả lời câu hỏi gốc |

Các nhánh hybrid dùng cùng ứng viên DBNet với cùng query. Xếp hạng theo tỷ lệ diện tích box chữ nằm
trong box SD-RPN, rồi theo diện tích giao và confidence. Ngưỡng khớp cố định là 30%. Khi thiếu ứng viên
đạt ngưỡng, hybrid dùng box SD-RPN. Khi thiếu box SD-RPN, hybrid quay về toàn ảnh. DBNet riêng quay
về toàn ảnh khi thiếu box chữ. Các box được nới 20% mỗi cạnh và cắt theo biên ảnh.

Chấm cùng 3.473 câu LR4 với A2. Hai ảnh dùng 288 + 288 visual token. Toàn ảnh và fallback dùng 576.
Thời gian đầu-cuối cộng cả hai selector ở nhánh hybrid. Chi phí DBNet báo theo một lượt detect ảnh
cho mỗi câu khi ước lượng độ trễ của từng nhánh. Job thực tế chia sẻ một lượt detect giữa các câu
cùng ảnh. TextOCR chỉ dùng hậu nghiệm để đo vùng chữ đáp án được giữ lại trên tập con khớp duy nhất
với ít nhất 8/10 đáp án. `summary.json` ghi cả kết quả trên tập con chung có crop. Kết quả nằm
trong `runs/dbnet-benchmark-…`, kèm JSONL từng câu và HTML xem ảnh.

### Chi phí phục hồi pixel và latent

```bash
python3 run_query_crop.py \
  --from-crop-run "$crop/runs/sdrpn-crop-20261107T173143Z-heum7_b7" \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" --arms S1b \
  --restoration-cost --profile-questions 32 \
  --pixel-model "$crop/local/fsrcnn/FSRCNN_x4.pb" \
  --pixel-runtime "$crop/local/fsrcnn/python"
```

Các nhánh dùng Qwen gốc với 576 visual token trên mỗi câu của `val_inner`. Có toàn ảnh gốc,
toàn ảnh phục hồi pixel FSRCNN ×4, toàn ảnh bật S1b, crop với hai trạng thái tắt/bật S1b,
và bốn trạng thái S1b khi ghép toàn ảnh + crop (288 + 288). SD-RPN chọn box theo query,
nới 20% và giữ quy tắc fallback của fair fusion. Nhánh crop tính cả chi phí SD-RPN ở câu fallback.
FSRCNN chạy CPU, nhận đúng ảnh giảm độ phân giải 4 lần trước bước bicubic upsample của LR4.
Checkpoint ghim SHA-256 trong code, chạy với `opencv-contrib-python-headless==4.10.0.84`
cài bằng `pip --no-deps --target` vào thư mục `--pixel-runtime` khi dùng lệnh `pip install`.
[Trọng số FSRCNN ×4](https://github.com/Saafke/FSRCNN_Tensorflow/tree/6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d/models)

`cost.csv` báo accuracy, visual token, prompt token, token sinh, latency trung bình, p50 và p95.
Thời gian tính theo từng bước chọn vùng, phục hồi pixel, chuẩn bị input, vision encoder, phục hồi
latent và sinh đáp án. Các bước CUDA đo có đồng bộ, sau khi nạp model, ở lượt chạy thường.
FLOPs ước tính trên cùng 32 câu chọn theo seed cố định ở lượt chạy riêng, dùng bộ đếm phép toán
đã đăng ký của PyTorch và `Net.getFLOPS` của OpenCV cho FSRCNN. Ước tính có giới hạn theo các
phép toán được từng bộ đếm hỗ trợ. Mỗi nhánh nhận chi phí cho các bước cần để trả lời riêng một
câu, dù job dùng lại kết quả giữa các nhánh hoặc các câu cùng ảnh. Dùng `--images 6` để chạy smoke.
Kết quả nằm trong `runs/restoration-cost-…`.

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

## Official validation follow-up

From the crop checkout on the login node, submit the two independent jobs with:

```bash
python3 run_followup.py --from-confirm runs/a2-fusion-official-20261110T115853Z-k5mhzwkw
```

Use `--mode restoration` or `--mode crop` to submit just one study.
Both workers use all 5,000 official TextVQA validation questions and the frozen
S1 checkpoints referenced by the completed confirmation run.

- Restoration evaluates the original audited LR4, Gaussian blur of CLEAN at
  Pillow radius 4 px, and Gaussian blur of the existing bicubic LR4 at the same
  radius. It writes separate accuracy and paired CSVs for each condition. Blur
  is generated in memory. Each condition contains A0, S1a, S1b, A1, A2, the
  two CLEAN-only controls, and S1b combined with A2.
- The selector benchmark compares full-image A2 to crop-only and full-plus-crop
  A2 for SD-RPN 3B, ViCrop rel-att on Qwen2.5-VL-3B, SD-RPN 7B, and Vision-RL²
  7B. It uses the existing 576-token reader preset and 20% padding per side.
  The 7B comparison uses the same single-box decoder for both checkpoints.
  ViCrop selects its attention layer on grounded val_inner cases before
  inference on official validation. TextOCR coverage is a diagnostic and
  fallback to full-image A2 remains in the 5,000-question denominator.
- The crop worker stages pinned public 7B assets in `local/visionrl2` (around
  21 GB of weights after deduplicating three identical shards). Optional
  pre-staging uses the existing Q-LRG environment:
  `python -m scripts.selector_assets local/visionrl2`. A file lock protects
  the cache. The audit covers released VisualCoT candidate and 7B RL pools,
  with incomplete coverage of upstream SD-RPN training data.
