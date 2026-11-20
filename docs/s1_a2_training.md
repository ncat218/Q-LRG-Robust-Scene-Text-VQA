# Huấn luyện phục hồi latent với A2 đóng băng

Thực hiện theo yêu cầu triển khai sau kết quả job 68268. Mục tiêu là kiểm tra
module phục hồi có học được cách cung cấp feature hữu ích cho A2 không, thay
vì ghép hai checkpoint được huấn luyện độc lập.

## Pipeline

```text
CLEAN ── cùng vùng ảnh ── Vision Encoder + Merger đóng băng ── feature teacher
LR4   ── full hoặc crop ─ Vision Encoder + Merger đóng băng ── module visual ── LLM + A2 ── đáp án
                                                              ↑
                                                     chỉ module này được cập nhật
```

A2 là LoRA Q/V trong LLM của job 11305, giữ nguyên trọng số. Qwen2.5-VL-3B,
vision encoder, Merger và Vision-RL² cũng giữ nguyên. Gradient được truyền
qua A2/LLM để cập nhật module visual, không cập nhật A2 hoặc LLM.

## Bốn biến thể trong cùng một job

| Đầu vào | Method | Mục tiêu học |
|---|---|---|
| Full | `restoration` | Loss đáp án qua A2 + MSE với feature CLEAN + regularization |
| Full | `ce_only` | Cùng module và regularization, nhưng hệ số MSE = 0 |
| Crop Vision-RL² | `restoration` | Loss đáp án qua A2 + MSE với feature CLEAN cùng crop + regularization |
| Crop Vision-RL² | `ce_only` | Cùng module và regularization, nhưng hệ số MSE = 0 |

`ce_only` là đối chứng spatial adapter có cùng kiến trúc, không phải A1 cũ
(A1 cũ không có các spatial block). Regularization residual và phạt residual
trên CLEAN giữ giống nhau giữa hai method. Feature CLEAN vẫn có thể được
tính để ghi chỉ số, nhưng không tác động gradient qua MSE trong `ce_only`.

Cả bốn module khởi tạo mới cùng seed 42, hidden 224, hai spatial block, đầu
ra delta khởi tạo bằng 0. Không lấy trọng số S1b mới từ job 68269 và không
cần đợi job đó xong. Chỉ A2 đã học của 11305 được tải vào reader. Job 68269
tiếp tục chạy từ snapshot cũ, không bị cập nhật code/config.

## Dữ liệu và crop

- Chọn checkpoint trên train_inner 19.758 ảnh / 31.129 câu, val_inner
  2.195 ảnh / 3.473 câu; tách theo image ID và kiểm SHA-256 / question ID.
- Crop được chọn bằng Vision-RL² 7B từ **LR4 + câu hỏi**, nới biên 20%.
  Cùng box cắt CLEAN và LR4; hai feature grid phải khớp. Không dùng đáp án
  hoặc TextOCR để chọn box. Không có ROI thì fallback toàn ảnh và vẫn giữ mẫu.
- Proposal chỉ sinh một lần cho cả hai method crop, lưu tọa độ/metadata,
  không lưu toàn bộ crop hay feature tensors xuống đĩa.
- Giữ CLEAN 40%, LR4 60% theo protocol 11305, cùng thứ tự mẫu và đáp án lấy
  mẫu giữa các method. Không thêm Blur trong job này.
- Chọn thời lượng trên val_inner rồi khởi tạo lại, refit đủ **21.953 ảnh /
  34.602 câu official train** theo số fractional epoch đã chọn. Khi refit,
  không dùng lại val_inner để early stopping vì nó đã trở thành dữ liệu train.
- Không chấm 5.000 câu official validation trong job train này.

## Cấu hình và checkpoint

AdamW LR 2e-4, effective batch 4, cùng loss weights gốc ngoài ablation MSE=0.
Mỗi method tối đa 8 epoch, minimum 2 trước khi dừng sớm; đánh giá mỗi 0,5
epoch, patience 4 lần, min_delta 0,1 điểm phần trăm. Chọn LR4 tốt nhất với
CLEAN không thấp hơn A2 baseline cùng view quá 1 điểm.

Một baseline A2 được tính và kiểm hash một lần cho mỗi view, dùng chung cho
cả hai method để tránh hai baseline khác nhau do nhiễu decode.

Mỗi biến thể chỉ giữ:

- `selection/best.pt`: checkpoint đủ điều kiện có VQA LR4 tốt nhất.
- `selection/latest.pt`: trạng thái model/optimizer/RNG để resume.
- `refit/latest.pt`: checkpoint cuối/resume của refit, nếu có.

Tối đa 12 file checkpoint cho bốn biến thể, không tạo file theo từng epoch.
Ghi đè nguyên tử có thể tạm cần thêm dung lượng một checkpoint. Không lưu
feature cache hàng trăm GB. Bản gốc 11305 và kết quả cũ được giữ nguyên.

## Kiểm chứng reader và đầu ra

- Kiểm provenance/SHA-256 A2, số lượt học và binding với source 11305.
- Kiểm prefill identity khi delta=0 và gradient qua A2 đóng băng trước khi
  đo baseline dài. Gradient phải tới module visual; không tới trọng số reader.
- Kiểm `requires_grad`, parameter gradients và digest trọng số A2 khi lưu
  định kỳ, sau validation và cuối mỗi giai đoạn. Optimizer chỉ chứa module visual.
- Log loss đáp án, feature loss, residual loss, gradient norm và peak VRAM.
- Mỗi lần validation lưu accuracy CLEAN/LR4, feature MSE, tỷ lệ norm residual,
  độ dài đáp án và các dự đoán best/latest để EDA lỗi đọc chữ/trả lời thừa.
- `progress.json` ghi view/method/stage đang chạy. Thứ tự: full restoration,
  full ce_only, crop restoration, crop ce_only; mỗi biến thể có selection/refit.
- `validation_comparisons.csv` so hai method và baseline với paired image
  bootstrap. Đây là phân tích val_inner sau chọn checkpoint, không phải kết quả
  xác nhận cuối; các method có thể được chọn ở số epoch khác nhau.

Walltime mỗi allocation tối đa 72 giờ theo QoS `mig2-35`. Bốn biến thể và
refit có thể cần nhiều hơn một allocation; resume cùng thư mục, không tạo
run mới hay submit trùng. Epoch cap/early stopping không chứng minh hội tụ
tuyệt đối. Chưa huấn luyện jointly LoRA trong workflow này.

## Lệnh

```bash
python3 run_s1_crop_study.py --mode s1_a2_convergence \
  --from-run "$OLD_CROP_RUN" \
  --train-config configs/s1_a2_convergence.json --view both \
  --selector-assets "$ORIGINAL_CROP_REPO/local/visionrl2" --time 3-00:00:00

python3 run_s1_crop_study.py --mode s1_a2_convergence \
  --resume "$TRAIN_RUN" --time 3-00:00:00
```

Đây là adaptation thử nghiệm Q-LRG, không phải tái hiện nguyên bài Latent
Denoising hoặc MAA. Căn cứ paper, khác biệt cơ chế và các giả thuyết được lưu
trong `docs/latent_restoration_study.md`, mục phối hợp phục hồi với A2.
