# Phục hồi latent có giám sát hành vi và phân biệt patch

Ngày thiết kế: 26/11/2026. Nghiên cứu tiếp Q-LRG theo yêu cầu, phạm vi chính
là **full image**; giữ nguyên job mixed crop 163061 đang chạy. Không sinh ảnh RGB.

## Sửa các kết luận quá mạnh trong bản phân tích được gửi

1. Job 112448 chứng minh các checkpoint crop-trained hiện tại **chưa vượt**
   S1b cũ một cách rõ ràng. Không đủ để loại trừ lệch miền full/crop trong mọi
   kiến trúc hoặc kết luận thông tin pixel là nút thắt duy nhất. A0 crop mới
   thực tế tăng +1,134 điểm [0,352; 1,932] so với không phục hồi ở native.
2. Điểm LR4 và CLEAN cùng tăng chỉ là dấu hiệu adaptation. Phải báo tương tác
   bốn điểm và CI theo cặp; không quy toàn bộ lợi ích cho adaptation từ hai
   số trung bình. MSE/cosine cũng không tự chứng minh khả năng đọc chữ.
3. `min_delta=0,1` là tham số patience, không phải ngưỡng ý nghĩa thống kê.
   So nó với CI của một phép so khác không chứng minh epoch được chọn ngẫu
   nhiên. Ngay cả CI epoch-epoch cũng chưa đo biến thiên train. Tách tác động
   của selection, refit, số bước và seed trước khi khẳng định nguyên nhân.
4. Bản phân tích nói A2 crop được chọn trên fusion là sai: selection của
   81950 là **crop-only** trên val_inner, refit rồi mới chấm official_val.
   Train qua A2 làm giảm một số tác hại khi ghép module, chưa chứng minh đã
   loại bỏ mọi vấn đề phối hợp. Nhiều CI chứa 0 không có nghĩa tương đương.
5. Crop-only thấp hơn fusion không đủ chứng minh thiếu context, vì token và
   tỷ lệ chữ có thể khác. Native fusion tăng token, fair fusion vẫn cần đọc
   CI và kiểm soát độ phóng đại khi kết luận cơ chế.
6. A2 giữ làm đối chứng adaptation, có thể tách bảng khỏi phương pháp phục
   hồi; không bỏ kết quả chỉ để tránh một đối chứng mạnh hơn. Oracle CLEAN
   dùng thông tin bổ sung khi test, nên khoảng cách oracle không phải lượng
   thông tin mà LR4-restorer chắc chắn có thể khôi phục.
7. Các số P0 6,8%/cosine 0,26, 50 slide và tình trạng phần Danh trong bản
   gửi không được xác minh lại ở lượt này; không dùng làm bằng chứng mới.
   Không thực hiện lời nhắc xóa file /tmp của một trợ lý khác.

## Nguồn đã đọc và thành phần thực sự lấy dùng

| Nguồn gốc | Paper chứng minh/đề xuất | Q-LRG lần này dùng gì, khác gì |
|---|---|---|
| [Hinton, Vinyals, Dean, 2015, Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Distillation bằng phân phối đầu ra mềm | Teacher/student cùng Qwen đóng băng; chỉ module visual cập nhật. Teacher nhận CLEAN, student nhận feature LR4 đã hiệu chỉnh. |
| [Wang et al., 2026, NOPD, §3, Eqs. 4–7](https://arxiv.org/html/2607.23125) | Bất đối xứng clean/noisy, teacher stop-gradient, on-policy reverse KL để post-train VLM | Lấy ý tưởng clean/noisy functional supervision. Dùng **forward KL trên gold prefix**, không rollout on-policy, không cập nhật LLM. Gate teacher trùng token GT là lựa chọn mới của Q-LRG. **Không gọi là tái hiện NOPD.** |
| [Parikh et al., 2026, Latent Denoising, §3.3 Eq. 11](https://arxiv.org/html/2604.21343v2) | Intra-image patch InfoNCE cùng reconstruction/relational losses, khôi phục teacher từ hidden state giữa LLM; có masking/noise trên latent | Lấy InfoNCE patch identity. Áp vào output restorer sau Merger trên LR4 thật; không có mid-LLM decoder/noise masking, không train LLM. Text-anchor và loại negative teacher cosine ≥0,98 là lựa chọn Q-LRG. **Không tái hiện toàn bài.** |
| [Li et al., MAA, ICML 2026](https://proceedings.mlr.press/v306/li26hs.html), [official code](https://github.com/acnul/MAA) | Frozen-vision layer adapters với paired distillation và adversarial token constraint | Dùng để phản biện MSE-only; **không** thêm GAN/discriminator trong job này. Checkpoint public là LLaVA-Mistral-7B, không cắm trực tiếp vào Qwen3B. |
| [Zhou et al., Uni-DocRobust, ICML 2026](https://proceedings.mlr.press/v306/zhou26aw.html) | Universal core tiền huấn luyện canonical space, multi-teacher và feature adapters | Chưa tái hiện core tiền huấn luyện. Không gọi một Transformer ngẫu nhiên hai tầng là bản tương đương paper. Metadata/abstract xác nhận; PDF raw không đọc được qua công cụ web lần này. |

Các nguồn gần đây khác đã kiểm abstract: [Distilling Vision Transformers for
Distortion-Robust Representation Learning](https://arxiv.org/abs/2604.22529)
dùng multi-level KD cho classification. Đây là hướng mở rộng backbone, chưa
phải bằng chứng phục hồi chữ LR4 hoặc thay đổi được triển khai trong job này.

## Giả thuyết và pipeline

```text
CLEAN → Vision + Merger frozen → F_clean → Qwen frozen → teacher logits (train only)
LR4   → Vision + Merger frozen → S1b → F_restored → Qwen frozen → short answer
                                  ↑ chỉ module này được học
```

Teacher và student nhận cùng query và cùng prefix đáp án huấn luyện, bao gồm
vị trí EOS. Inference chỉ nhận ảnh suy giảm và query. Không dùng ảnh CLEAN,
GT, TextOCR hoặc teacher logits trong dự đoán; TextOCR ở evaluation chỉ ghi
diagnostic theo vùng chữ, không đi vào module/LLM.

- **continued:** train tiếp S1b với CE + MSE + cùng regularization; đối chứng
  bắt buộc cho ảnh hưởng của thêm bước train.
- **answer_kd (ứng viên 1):** continued + `0,5 × T² KL(p_clean || p_restored)`,
  `T=2`, chỉ trên vị trí teacher argmax trùng token GT. Giả thuyết: khớp hành
  vi trả lời giúp feature correction hữu ích hơn, kể cả điểm dừng EOS.
- **answer_kd_patch (ứng viên 2):** answer_kd + `0,1 × intra-image InfoNCE`,
  temperature 0,1. Positive là CLEAN token đúng vị trí; tối đa 128 anchor ưu
  tiên text cells, keys là teacher tokens trong cùng ảnh. Loại các teacher
  negative gần trùng (cosine ≥0,98) để tránh ép các vùng lặp giống nhau tách xa.
  Giả thuyết: tăng tính phân biệt patch, thay vì chỉ kéo về feature trung bình.

MSE gốc vẫn giữ hệ số 1; residual 0,01, thêm CLEAN residual 0,1 như 11305.
Ngưỡng và trọng số bổ sung là lựa chọn thử nghiệm Q-LRG, chưa được tối ưu.
Không gọi loss patch là loss đọc ký tự. CER/transcription là phép đo thêm cần
làm nếu muốn khẳng định phục hồi chữ, không chỉ VQA.

## Công bằng và ngân sách

- Cả ba arm khởi tạo **cùng trọng số S1b 11305**, không dùng bộ mới đang thua.
- Fresh AdamW giống nhau, LR **5e-5**, weight decay 0,01, accumulate 4,
  seed 42, clip norm 1; không nạp optimizer momentum cũ cho riêng control.
- Đủ **19.758 ảnh / 31.129 câu train_inner**, thêm **2 epoch / 62.258 lượt
  câu mỗi arm**, cùng thứ tự ảnh, CLEAN40%/LR4 và đáp án được lấy mẫu.
- Tách **2.195 ảnh / 3.473 câu val_inner**. Theo dõi tại 1 và 2 epoch, không
  early stopping, không chọn epoch thắng 0,1 điểm. Endpoint chính khóa ở
  **2 epoch thêm**; không refit gộp val_inner trong lượt này.
- Khớp số bước và số tham số; KD có thêm teacher compute nên **không** khớp
  GPU-hours. Báo thêm chi phí thay vì giả định hai loại fairness giống nhau.
- Chỉ train trên CLEAN/LR4; Blur không đưa vào train. Một seed là screening
  trên toàn bộ train_inner, chưa kết luận phương pháp đúng/sai tổng quát.
- Giữ tối đa **latest.pt + epoch1.pt mỗi arm**, tổng 6 file, không lưu feature
  cache hàng trăm GB. Khoảng 5 lần ghi định kỳ cộng các lần ở epoch/validation,
  ghi đè nguyên tử; dung lượng tạm có thể thêm một checkpoint.

## GPU gates và kết quả sau train

Trước bất kỳ optimizer step dài nào, job thử cả ba objective: identity
injection, causal prefix/query/EOS, gradient CE đến restorer, gradient KD khi
teacher có vị trí hợp lệ, gradient contrast khi có negative hợp lệ. Reader
không có gradient. Ghi gradient norm từng thành phần để xem loss nào lấn át.

Sau cả ba train, đánh giá các checkpoint cố định trên **5.000 câu / 3.166 ảnh**:

- A0, S1b 11305, ba arm mới; cùng các phiên bản qua A2 đóng băng.
- **CLEAN, LR4, BLUR4, LR4_BLUR4** báo riêng; Gaussian Blur Pillow radius4.
  Đây là blur cross-degradation, không gọi là đã train phục hồi blur.
- Accuracy, paired image-cluster CI 95% / 10.000 resamples, đúng→sai/sai→đúng,
  hiệu ứng `(ΔLR4 − ΔCLEAN)` kèm CI, text/background feature MSE, cosine hướng
  hiệu chỉnh và độ lớn residual. MSE/cosine không thay thế VQA/CER.
- Native full image, cùng token protocol giữa arm; không crop/selector/token
  pruning ở job này. Khoảng 200.000 dự đoán cuối, cộng validation monitoring.
- p50/p95 reader latency và peak CUDA; registered FLOPs trên 3 ảnh/8 decode,
  không gọi là toàn bộ FLOPs chính xác nếu operator chưa được hỗ trợ.

Chấp nhận đầu tư tiếp khi candidate vượt **continued** cùng exposure với CI
dương trên LR4, CLEAN không giảm quá 1 điểm và có diagnostic phù hợp. So với
A2 giữ nguyên để đánh giá thực dụng; chưa vượt A2 vẫn báo trung thực. CI theo
checkpoint không đo seed variance, chưa hiệu chỉnh multiple comparisons.
Mỗi nguồn/checkpoint/protocol/citation được lưu hash trong run binding.

Một job tuần tự, MIG H200 2g.35gb, CPU8/RAM48GB, walltime tối đa72h. Ước
lượng thô **40–65 giờ**, KD tăng compute; nếu timeout thì resume checkpoint
trong cùng run, không hạ lượng dữ liệu/epoch âm thầm. Job 163061 giữ nguyên.

```bash
python3 run_semantic_restoration.py --from-matrix /absolute/path/to/completed/crop_matrix
python3 run_semantic_restoration.py --resume /absolute/path/to/semantic-restoration-run
```
