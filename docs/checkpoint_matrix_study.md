# So sánh checkpoint cũ và mới trên cùng tập đánh giá

Yêu cầu ngày 23/11/2026: chấm Qwen gốc, S1b ban đầu của job 11305,
A2 và module mới cho full/crop/fusion. Đây là job **inference**, không train thêm.

## Phạm vi khóa trước khi chấm

- Mỗi cấu hình chấm đủ **5.000 câu / 3.166 ảnh official_val**. Đây là validation
  chính thức đã được dùng trong dự án, không phải tập test chưa từng nhìn.
- CLEAN và LR4, hai bảng riêng. CLEAN dùng cùng box được chọn từ LR4 + query,
  không chạy lại selector trên CLEAN, không dùng đáp án hay TextOCR để chọn box.
- Vision-RL² 7B, box và padding 20% của benchmark cũ được kiểm checksum và tái sử dụng.
- Fair: tổng **576 visual token**; fusion **288 + 288**. Khi không có ROI,
  fallback một toàn ảnh 576 token, vẫn giữ nguyên câu trong cohort.
- Native: kích thước theo processor đã khóa; full/crop/fusion có thể khác số token.
- Ghi mọi token của crop vào reader, không lọc thêm token chữ.

## Các profile

| Profile | Module full/crop | Reader đánh giá |
|---|---|---|
| `old11305` | Cùng S1b full-trained của 11305 cho cả hai view | A0 và A2 |
| `new_A0` | Hai module riêng, train full/crop qua Qwen gốc, chọn thời lượng rồi refit toàn bộ train | A0 và A2 |
| `new_A2_restoration` | Hai module train riêng qua A2 đóng băng, loss đáp án + khớp feature CLEAN | A2 |
| `new_A2_ce_only` | Cùng kiến trúc nhưng hệ số loss khớp feature bằng 0 | A2 |

Trong từng profile: full tắt/bật, crop tắt/bật, fusion tắt cả hai / bật full /
bật crop / bật cả hai. Profile là nguồn checkpoint; A0/A2 là reader trả lời.
`new_A0` qua A2 đo việc ghép module train qua A0 với A2; nó không trở thành
module được train qua A2 chỉ vì reader lúc đánh giá là A2.

Mỗi nguồn mới phải có COMPLETE cấp group và view/method; checksum summary,
selection và refit phải khớp. Chỉ dùng checkpoint refit đã hoàn tất, không
lấy latest đang train hoặc chọn checkpoint theo official_val. Nếu selection
giữ baseline khởi tạo zero, profile ghi rõ `baseline_kept`; không gọi đó là
module phục hồi đã train thắng baseline. Những cấu hình giống hoàn toàn được
alias về một generation chung. Baseline/11305 được **chấm lại cùng job**,
không trộn dự đoán của job cũ vào các hiệu ứng của checkpoint mới.

Nếu hai module full qua A2 giữ baseline còn hai module crop có checkpoint
mới, kế hoạch gồm **120 cấu hình duy nhất × 5.000 câu = 600.000 dự đoán**
(tính cả CLEAN/LR4), trình bày thành 192 dòng profile. Số generation thực tế
được khóa tự động theo kết quả lựa chọn checkpoint, không tự ý bỏ arm.

## Độ đo và artifact

- TextVQA MMF soft accuracy, scorer pin revision như các job trước.
- Paired bootstrap 10.000 lượt theo cụm image ID; mỗi cặp giữ cùng condition,
  token budget và 5.000 câu. CI theo checkpoint cố định, không đo biến thiên
  seed và chưa hiệu chỉnh multiple comparisons.
- So từng nhánh với full không phục hồi, cùng view không phục hồi, 11305 cùng
  cách bật/tắt; thêm module train qua A0 so với qua A2 và feature-loss so CE-only.
- `comparison_table.csv`: bảng đầy đủ profile/reader/condition/budget/layout/toggle.
- `accuracy.csv`, `paired.csv`, `error_cases.csv`: điểm, CI, các câu đổi điểm
  và đúng↔sai. Nguyên nhân để `unreviewed` cho đến khi xem ảnh.
- `flops.csv`: FLOPs các operator PyTorch đăng ký được, gồm vision/restorer/LLM,
  cố định 8 token decode, đo riêng trên 3 ảnh; không bao gồm selector và những
  operator chưa hỗ trợ. Không gọi đây là tổng FLOPs hoàn chỉnh của pipeline.
- Token thật từng view; peak CUDA VRAM; thời gian p50/p95. Vision/restorer được
  tái dùng để tránh tính lại, thời gian selector lấy từ archive: **không phải
  đo lại latency end-to-end**. Native fusion nhiều token hơn full riêng.

## Thực thi và an toàn dữ liệu

Worker chạy từ snapshot riêng, không sửa hai snapshot đang train, không đổi
checkpoint 11305, không sửa/xóa workspace Danh. Không lưu feature cache, không
lưu ảnh review hàng loạt và không tạo checkpoint mới. Chỉ lưu dự đoán/metadata
theo ảnh để resume; dự trù vài GB output, không đảm bảo quota chỉ từ `df`.

Một job Slurm, 1 MIG H200 2g.35gb, 8 CPU, 48 GB RAM, time limit 48 giờ
(ước lượng thô khoảng 20–30 giờ từ throughput job 68268, không phải cam kết).
Dependency `afterok:81950` bảo đảm nguồn A2 hoàn tất trước khi đọc checkpoint.
Nếu train nguồn failed/time-out, job đánh giá vẫn chờ dependency; phải xử lý
nguồn trước, không lấy checkpoint đang ghi để chạy bù.

```bash
python3 run_checkpoint_matrix.py \
  --from-matrix /absolute/path/to/completed/crop_matrix \
  --a0-training /absolute/path/to/s1_convergence \
  --a2-training /absolute/path/to/s1_a2_convergence \
  --dependency 81950 --time 2-00:00:00

# Resume cùng snapshot, sau khi job trước đã dừng:
python3 run_checkpoint_matrix.py --resume /absolute/path/to/checkpoint-matrix-run
```

Launcher kiểm job trùng theo cùng hai nguồn train trong snapshot. Worker khóa
thư mục run và binding checkpoint/config; shard khác binding bị từ chối.
Nếu failed, journal lỗi nhỏ được ghi thêm dưới `~/.qlrg_recovery/<run-name>/`.
