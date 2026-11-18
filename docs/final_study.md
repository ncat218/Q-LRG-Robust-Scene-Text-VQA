# Workflow chốt thực nghiệm Q-LRG

Mục tiêu là đánh giá phục hồi latent toàn ảnh trên TextVQA LR4 và so vị trí
trước/sau Merger. Bốn cấu hình được chọn trước khi xem bảng điểm validation
chính thức của lượt này.

| Cấu hình | Cơ chế | Nguồn checkpoint |
| --- | --- | --- |
| A0 | Qwen2.5-VL-3B đóng băng | Model revision trong `configs/model_qwen.yaml` |
| S1b | Residual không gian sau Merger, học từ feature CLEAN và đáp án | Job 11305, `s1/variants/S1b/checkpoint.pt` |
| Pre | Residual trên lưới patch trước Merger, dùng loss và optimizer của 11305 | Job 48633, `pre.pt` |
| A2 | LoRA trên attention của LLM, học trả lời | Job 11305, `s1/variants/A2/checkpoint.pt` |

S1b, Pre và A2 dùng 19.758 ảnh train_inner, hai epoch, 62.258 lượt câu hỏi và
15.565 bước cập nhật ở seed 42. Pre và S1b có khoảng 1,03 triệu tham số
trainable. A2 là đối chứng thích nghi LLM và cần báo số tham số riêng.

## Trình tự

1. Xác thực job 48633 hoàn tất và đọc các chênh lệch Pre/S1b/A0 trên val_inner.
2. Chấm lại cả bốn checkpoint trên cùng 5.000 câu / 3.166 ảnh official_val,
   cho cả CLEAN và LR4. Launcher có thể đặt `afterok:48633` để đợi train xong.
3. Báo TextVQA MMF soft accuracy và CI 95% theo cặp, bootstrap 10.000 lần
   theo image ID. Giữ câu hỏi của cùng ảnh trong một cụm.
4. Báo delta(LR4) trừ delta(CLEAN) của mỗi nhánh so với A0. Đọc kết quả cùng
   mức tăng trên từng điều kiện để đánh giá lợi ích đặc thù với suy giảm.
5. Đọc bảng ca tăng/giảm điểm và kiểm chứng thủ công lý do lỗi. Dùng demo để
   xem ảnh đầu vào, câu hỏi, 10 đáp án tham chiếu và dự đoán của bốn cấu hình.
6. Chọn kết luận dựa trên độ chính xác, điểm CLEAN, chi phí và các ca lỗi.
   Nếu chênh lệch nhỏ, xác nhận thêm seed khi ngân sách cho phép.

Official validation đã xuất hiện trong các thực nghiệm trước của dự án.
Lượt chấm này là so sánh checkpoint được khóa, cần ghi lịch sử sử dụng tập
validation khi báo cáo. Một seed chưa đo được biến thiên giữa các lần train.

## Chạy trên HPC

Các đường dẫn dưới đây là vị trí của các artifact đã có trên UIT-HPC.

```bash
main=/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA
crop=/datastore/dungmt/textvqa/Tuan/Q-LRG-crop

python3 run_final_study.py \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" \
  --pre-run "$crop/runs/pre-11305-pWZx2Ojh" \
  --env "$main/.venv" --paths "$main/configs/paths_hpc.yaml" \
  --split official_val --after-job 48633 --submit
```

Launcher tạo bản sao code riêng cho lượt đánh giá và ghi Job ID. Worker lưu
kết quả theo ảnh để resume sau gián đoạn, kiểm tra manifest chung cho tất cả
nhánh, và xác thực nguồn cùng số bước của checkpoint.

```bash
python3 run_final_study.py --resume /absolute/path/to/evaluation --submit
```

## Tệp dùng để đánh giá

| Tệp | Nội dung |
| --- | --- |
| `accuracy.csv` | Điểm của từng cấu hình trên CLEAN và LR4, cùng số ảnh/câu |
| `paired.csv` | Chênh lệch và CI 95%, số câu tăng/giảm điểm |
| `degradation_interaction.csv` | delta(LR4) trừ delta(CLEAN), kèm CI |
| `predictions.jsonl` | Dự đoán và điểm từng câu |
| `error_cases.csv` | Các cặp câu để EDA, cột lý do để người đọc kiểm chứng |
| `review_clean.html`, `review_lr4.html` | Lọc và đối chiếu toàn bộ dự đoán |
| `demo.html`, `demo_cases.json` | Phát lại tối đa 24 ca được chọn cân bằng theo các hướng tăng/giảm |
| `cost.csv`, `cost_summary.csv` | Chi phí suy luận theo ảnh và số tổng hợp |
| `run.json`, `COMPLETE.json` | Checkpoint, hash dữ liệu, phiên bản model và checksum output |

Demo HTML hiển thị ảnh CLEAN/LR4 đầu vào. Module phục hồi sửa vector feature.
Các proxy như đáp án chứa số hoặc đáp án dài hơn khi điểm giảm dùng để chọn
ca xem xét. Cột `manual_reason` để trống cho đến khi kiểm chứng từng ảnh.

Chi phí được đo trên cùng 40 ảnh LR4, batch size 1, hai ảnh warm-up và thứ tự
cấu hình luân phiên. Báo p50/p95 encode + phục hồi theo ảnh và toàn request
theo câu hỏi. FLOPs đếm trên 10 ảnh với tám bước decode cố định, chỉ bao gồm
operator được bộ đếm PyTorch hỗ trợ. Peak allocated VRAM là chi phí inference
khi cả ba adapter cùng nằm trên GPU. VRAM train lấy từ các log train tương ứng.

## Demo suy luận với ảnh mới

Trong một allocation GPU đang có, dùng Python của môi trường HPC và code của
thư mục đánh giá để giữ nguyên phiên bản. Lệnh sau sinh dự đoán CLEAN/LR4 của
bốn cấu hình. Khi ảnh gốc đã suy giảm, CLEAN chỉ biểu thị chính ảnh được nạp.

```bash
cd /absolute/path/to/evaluation/code
"$main/.venv/bin/python" -B run_final_demo.py \
  --evaluation /absolute/path/to/evaluation \
  --image /absolute/path/to/image.jpg \
  --question "What number is written on the sign?" \
  --output /absolute/path/to/new-demo-result.json
```

Script demo dùng checkpoint được khóa và yêu cầu allocation GPU.
