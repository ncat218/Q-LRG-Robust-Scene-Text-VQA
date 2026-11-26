# Full phục hồi bằng S1b 11305 + crop phục hồi bằng module train riêng crop

Yêu cầu ngày 26/11/2026: chấm tổ hợp còn thiếu sau job 112448. Đây là một
job **inference**, không train thêm và không thay checkpoint.

## Các checkpoint cố định

| Profile | Module full | Module crop | Reader |
|---|---|---|---|
| `old11305` | S1b 11305 | Cũng S1b 11305 | Qwen gốc và A2 |
| `mixed_old_full_A0_crop` | S1b 11305 | Crop-refit job 81949, train qua Qwen gốc | Qwen gốc và A2 |
| `mixed_old_full_A2_crop` | S1b 11305 | Crop-restoration-refit job 81950, train qua A2 đóng băng | A2 |

Mỗi profile có: full không phục hồi, full phục hồi; fusion tắt cả hai, chỉ
phục hồi full, chỉ phục hồi crop, phục hồi cả hai. Full dùng S1b 11305 trong
**mọi** profile; không dùng nhầm S1b-full mới của 81949.

Mọi cấu hình chấm cùng **5.000 câu / 3.166 ảnh official_val**, đủ CLEAN/LR4:

- Fair: tổng **576 visual token**, fusion 288 + 288.
- Native: processor giữ preset native đã khóa, không ép tổng token bằng full.
  Vẫn có giới hạn pixels/token của model; không phải token vô hạn.
- Vision-RL² 7B chọn box trên LR4 + query, nới 20%; tái dùng cùng box cho
  CLEAN/LR4, kiểm checksum nguồn. Không dùng TextOCR/đáp án chọn crop.
- Không có ROI thì fallback full nhưng giữ nguyên câu trong mẫu số; fusion
  fallback chỉ áp dụng module full, không áp module crop lên full.

Có **72 cấu hình duy nhất, 360.000 dự đoán**. Baseline giống nhau được alias;
bảng profile có 120 dòng. Baseline và old/old được chấm lại trong cùng job để
không suy hiệu ứng nhỏ từ hai lượt chạy khác nhau. Không chạy lại toàn bộ
ma trận 120 cấu hình của 112448; không chạy crop-only vốn đã có trong job đó.

## Đối chiếu chính, khóa trước khi chạy

1. Mixed cả hai bật so với **old/old cả hai bật**, cùng reader/condition/budget:
   crop-trained module có thay thế S1b cũ tốt hơn khi full vẫn dùng S1b cũ không?
2. Mixed cả hai bật so với **chỉ full bật**: thêm phục hồi crop có lợi không?
3. Mixed cả hai bật so với **chỉ crop bật**: thêm phục hồi full có lợi không?
4. Mỗi fusion so với full không phục hồi và fusion tắt cả hai.
5. Trên A2: crop module train qua A2 so với crop module train qua Qwen gốc,
   khi cùng giữ full S1b 11305. Qwen/A2 đều giữ nguyên trọng số.

Accuracy và paired bootstrap 95% theo image ID, 10.000 resamples. CI điều kiện
trên các checkpoint, chưa đo biến thiên seed hoặc hiệu chỉnh multiple testing.
Không chọn checkpoint theo điểm official_val; tập này đã dùng trong dự án.

Ghi token thật, p50/p95 thời gian, peak CUDA VRAM, registered reader FLOPs
trên 3 ảnh/8 decode tokens. Tái dùng timing vision/restorer và selector từ
archive, nên không gọi là đo lại latency end-to-end; FLOPs không gồm selector
và operator chưa hỗ trợ.

## Thực thi

`run_checkpoint_matrix.py --profile-set mixed` dùng worker đã kiểm chứng của
112448, thêm profile kết hợp hai checkpoint khác nhau. Worker kiểm binding,
checksum, trọng số reader và cohort, resume theo shard ảnh, không lưu feature
cache/ảnh review/checkpoint mới. Snapshot riêng, không sửa dữ liệu của Danh.

```bash
python3 run_checkpoint_matrix.py --profile-set mixed \
  --from-matrix /absolute/path/to/crop_matrix \
  --a0-training /absolute/path/to/s1_convergence \
  --a2-training /absolute/path/to/s1_a2_convergence \
  --time 1-00:00:00
```

Nguồn train đều đã hoàn tất nên không cần dependency. Một MIG H200 35 GB,
8 CPU / 48 GB RAM, time limit 24 giờ. Ước lượng thô khoảng 12–16 giờ từ job
112448, không phải cam kết thời gian. CPU unit test kiểm riêng tổ hợp module
old-full/new-crop và các cặp CI cần thiết trước khi submit.

## Trạng thái chuẩn bị ngày 26/11/2026

Code commit `3a10bfd` đã push `crop-branch`; 6 CPU unit tests về matrix đạt
trên máy phát triển. **Chưa submit Slurm, chưa có Job ID.** Ba lần kết nối
UIT-HPC đều timeout. Lần chạy SSH verbose xác nhận public-key authentication
thành công, nhưng server không phản hồi lúc mở session để chạy lệnh. Chưa
chạy được bộ test HPC/preflight checkpoint cho snapshot mới này. Không sửa
run/checkpoint cũ và không tạo job trùng khi kết nối chưa thành công.
