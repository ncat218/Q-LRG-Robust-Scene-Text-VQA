# Kết quả train và đánh giá checkpoint phục hồi latent — kiểm tra 26/11/2026

**Cả ba job đã hoàn tất, exit 0:0.** Các bảng dưới chỉ lấy từ job 112448; không trộn điểm từ job 68268 hoặc điểm val_inner.

| Job | Nội dung | Thời gian | Hoàn tất (thời gian Slurm) |
|---|---|---|---|
| 81949 | Module full/crop train qua Qwen gốc; selection rồi refit | 36:45:11 | 22/11/2026 20:13:10 |
| 81950 | Module qua A2 đóng băng; restoration và CE-only | 53:20:59 | 23/11/2026 12:48:58 |
| 112448 | Ma trận checkpoint cũ/mới, full/crop/fusion | 22:07:07 | 24/11/2026 10:56:05 |

## Kiểm chứng

Đã xác minh checksum 12 artifact trên HPC, binding run và manifest; đọc và chấm lại toàn bộ **600.000 dự đoán** bằng MMF scorer đã pin. Cả **120 cấu hình duy nhất** có cùng thứ tự **5.000 câu / 3.166 ảnh**; 192 dòng trình bày có alias baseline. Accuracy tính lại khớp báo cáo; mọi fair arm đúng 576 visual token; 24 identity gates khớp prefill logits tuyệt đối.

## Chú thích và checkpoint

- **A0 / Qwen gốc:** Qwen2.5-VL-3B-Instruct đóng băng, không LoRA.
- **A2:** LoRA Q/V attention của LLM từ job 11305; không phải module phục hồi feature. A2 giữ đóng băng trong hai job mới.
- **S1b cũ:** phục hồi residual sau Merger, học feature CLEAN và đáp án qua A0, checkpoint 11305; cùng module áp dụng full và crop.
- **S1b mới / new_A0:** hai module riêng train full/crop qua A0; checkpoint được chọn trên val_inner rồi khởi tạo lại để refit toàn bộ official train.
- **Phục hồi qua A2 / new_A2_restoration:** cùng họ kiến trúc visual, học đáp án qua A2 đóng băng và khớp feature CLEAN.
- **CE-only / new_A2_ce_only:** cùng module và regularization, không loss khớp feature CLEAN; đây là đối chứng adaptation visual.
- Crop: Vision-RL² 7B, chọn từ LR4 + query, padding 20%; không dùng đáp án/TextOCR chọn vùng. CLEAN cũng dùng cùng box LR4. Mọi câu vẫn được giữ, gồm 10 câu fallback full khi không có ROI.

| Module mới | Thời lượng chọn trên train_inner | Refit trên official_train | Ghi chú |
|---|---:|---:|---|
| new_A0_crop | 3.000 epoch | 3.000 epoch | 103,804 lượt câu hỏi |
| new_A0_full | 1.000 epoch | 1.000 epoch | 34,600 lượt câu hỏi |
| new_A2_ce_only_crop | 0.500 epoch | 0.500 epoch | 17,300 lượt câu hỏi |
| new_A2_ce_only_full | 0 | Bỏ qua | Giữ baseline zero; không phải checkpoint train thắng A2 |
| new_A2_restoration_crop | 2.500 epoch | 2.500 epoch | 86,504 lượt câu hỏi |
| new_A2_restoration_full | 0 | Bỏ qua | Giữ baseline zero; không phải checkpoint train thắng A2 |

Selection dùng 19.758 ảnh / 31.129 câu train_inner và 2.195 ảnh / 3.473 câu val_inner. Refit dùng 21.953 ảnh / 34.602 câu official train; official_val không dùng chọn checkpoint trong workflow này. Bản cũ 11305 dùng khoảng 2 epoch train_inner. Vì lượng dữ liệu và số bước khác nhau, old-vs-new là so checkpoint/workflow; không cô lập riêng tác động early stopping hay crop-training.

## Bảng kết quả đầy đủ

Accuracy là TextVQA soft accuracy (%). Fair dùng tổng 576 token; fusion 288+288, hoặc full 576 nếu fallback. Native trung bình full 929,42 / crop 393,40 / fusion 1.320,99 token, nên so layout native không khớp ngân sách.

### LR4 — fair576 — reader A0

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 |
|---|---:|---:|
| Toàn ảnh, không phục hồi | 62.518% | 62.518% |
| Toàn ảnh, bật module | 64.016% | 63.414% |
| Crop, không phục hồi | 61.198% | 61.198% |
| Crop, bật module | 62.378% | 62.464% |
| Full + crop, tắt cả hai | 63.910% | 63.910% |
| Full + crop, chỉ bật full | 65.240% | 64.894% |
| Full + crop, chỉ bật crop | 64.938% | 64.628% |
| Full + crop, bật cả hai | 65.174% | 64.800% |

### LR4 — fair576 — reader A2

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 | Module train qua A2 | CE-only qua A2 |
|---|---:|---:|---:|---:|
| Toàn ảnh, không phục hồi | 65.134% | 65.134% | 65.134% | 65.134% |
| Toàn ảnh, bật module | 64.506% | 64.184% | 65.134% | 65.134% |
| Crop, không phục hồi | 63.222% | 63.222% | 63.222% | 63.222% |
| Crop, bật module | 62.794% | 63.236% | 63.080% | 62.670% |
| Full + crop, tắt cả hai | 65.960% | 65.960% | 65.960% | 65.960% |
| Full + crop, chỉ bật full | 66.008% | 65.624% | 65.960% | 65.960% |
| Full + crop, chỉ bật crop | 65.860% | 65.764% | 66.040% | 65.916% |
| Full + crop, bật cả hai | 65.640% | 65.250% | 66.040% | 65.916% |

Hai profile train qua A2 giữ module full bằng zero do không có checkpoint train vượt baseline trên val_inner. Vì vậy các dòng bật full không thay đổi dự đoán; dòng bật cả hai bằng dòng chỉ bật crop.

### LR4 — native — reader A0

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 |
|---|---:|---:|
| Toàn ảnh, không phục hồi | 63.806% | 63.806% |
| Toàn ảnh, bật module | 64.900% | 64.042% |
| Crop, không phục hồi | 62.250% | 62.250% |
| Crop, bật module | 63.838% | 63.384% |
| Full + crop, tắt cả hai | 65.780% | 65.780% |
| Full + crop, chỉ bật full | 67.006% | 66.542% |
| Full + crop, chỉ bật crop | 66.698% | 66.152% |
| Full + crop, bật cả hai | 66.928% | 66.080% |

### LR4 — native — reader A2

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 | Module train qua A2 | CE-only qua A2 |
|---|---:|---:|---:|---:|
| Toàn ảnh, không phục hồi | 65.132% | 65.132% | 65.132% | 65.132% |
| Toàn ảnh, bật module | 64.330% | 63.612% | 65.132% | 65.132% |
| Crop, không phục hồi | 64.598% | 64.598% | 64.598% | 64.598% |
| Crop, bật module | 64.172% | 63.940% | 64.284% | 63.876% |
| Full + crop, tắt cả hai | 67.356% | 67.356% | 67.356% | 67.356% |
| Full + crop, chỉ bật full | 66.958% | 66.412% | 67.356% | 67.356% |
| Full + crop, chỉ bật crop | 67.318% | 66.900% | 67.512% | 67.350% |
| Full + crop, bật cả hai | 66.514% | 66.092% | 67.512% | 67.350% |

Hai profile train qua A2 giữ module full bằng zero do không có checkpoint train vượt baseline trên val_inner. Vì vậy các dòng bật full không thay đổi dự đoán; dòng bật cả hai bằng dòng chỉ bật crop.

### CLEAN — fair576 — reader A0

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 |
|---|---:|---:|
| Toàn ảnh, không phục hồi | 75.836% | 75.836% |
| Toàn ảnh, bật module | 76.916% | 76.044% |
| Crop, không phục hồi | 78.248% | 78.248% |
| Crop, bật module | 78.786% | 78.866% |
| Full + crop, tắt cả hai | 77.506% | 77.506% |
| Full + crop, chỉ bật full | 78.946% | 78.360% |
| Full + crop, chỉ bật crop | 78.486% | 78.316% |
| Full + crop, bật cả hai | 78.854% | 78.880% |

### CLEAN — fair576 — reader A2

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 | Module train qua A2 | CE-only qua A2 |
|---|---:|---:|---:|---:|
| Toàn ảnh, không phục hồi | 77.906% | 77.906% | 77.906% | 77.906% |
| Toàn ảnh, bật module | 77.000% | 76.230% | 77.906% | 77.906% |
| Crop, không phục hồi | 79.882% | 79.882% | 79.882% | 79.882% |
| Crop, bật module | 79.254% | 79.468% | 79.966% | 79.458% |
| Full + crop, tắt cả hai | 79.218% | 79.218% | 79.218% | 79.218% |
| Full + crop, chỉ bật full | 78.906% | 78.436% | 79.218% | 79.218% |
| Full + crop, chỉ bật crop | 78.966% | 78.750% | 79.250% | 79.198% |
| Full + crop, bật cả hai | 78.618% | 77.736% | 79.250% | 79.198% |

Hai profile train qua A2 giữ module full bằng zero do không có checkpoint train vượt baseline trên val_inner. Vì vậy các dòng bật full không thay đổi dự đoán; dòng bật cả hai bằng dòng chỉ bật crop.

### CLEAN — native — reader A0

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 |
|---|---:|---:|
| Toàn ảnh, không phục hồi | 79.118% | 79.118% |
| Toàn ảnh, bật module | 80.124% | 79.464% |
| Crop, không phục hồi | 77.306% | 77.306% |
| Crop, bật module | 78.418% | 78.536% |
| Full + crop, tắt cả hai | 80.966% | 80.966% |
| Full + crop, chỉ bật full | 81.486% | 81.154% |
| Full + crop, chỉ bật crop | 81.382% | 81.458% |
| Full + crop, bật cả hai | 81.304% | 81.204% |

### CLEAN — native — reader A2

| Đầu vào / cách bật | S1b cũ 11305 | S1b mới train qua A0 | Module train qua A2 | CE-only qua A2 |
|---|---:|---:|---:|---:|
| Toàn ảnh, không phục hồi | 80.902% | 80.902% | 80.902% | 80.902% |
| Toàn ảnh, bật module | 79.726% | 78.990% | 80.902% | 80.902% |
| Crop, không phục hồi | 79.118% | 79.118% | 79.118% | 79.118% |
| Crop, bật module | 78.406% | 78.664% | 78.872% | 78.788% |
| Full + crop, tắt cả hai | 82.004% | 82.004% | 82.004% | 82.004% |
| Full + crop, chỉ bật full | 81.138% | 80.832% | 82.004% | 82.004% |
| Full + crop, chỉ bật crop | 81.632% | 81.360% | 82.120% | 81.842% |
| Full + crop, bật cả hai | 80.828% | 80.494% | 82.120% | 81.842% |

Hai profile train qua A2 giữ module full bằng zero do không có checkpoint train vượt baseline trên val_inner. Vì vậy các dòng bật full không thay đổi dự đoán; dòng bật cả hai bằng dòng chỉ bật crop.

## Các chênh lệch chính trên LR4 và CI 95%

CI bootstrap 10.000 lần theo cụm image ID, cùng 5.000 câu; điều kiện trên các checkpoint cố định, không bao gồm biến thiên seed và không hiệu chỉnh nhiều phép so.

| So sánh (target trừ reference) | Budget | Delta điểm % | CI 95% |
|---|---|---:|---|
| A0 full: S1b cũ trừ không phục hồi | fair576 | +1.498 | [+0.733; +2.269] |
| A0 full: S1b mới trừ S1b cũ | fair576 | -0.602 | [-1.158; -0.065] |
| A0 crop: S1b cũ trừ không phục hồi | fair576 | +1.180 | [+0.395; +1.925] |
| A0 crop: S1b mới trừ S1b cũ | fair576 | +0.086 | [-0.465; +0.646] |
| A2 fusion: phục hồi crop qua A2 trừ không phục hồi | fair576 | +0.080 | [-0.293; +0.451] |
| A2 crop: phục hồi qua A2 trừ CE-only | fair576 | +0.410 | [-0.112; +0.939] |
| A2 fusion: phục hồi qua A2 trừ CE-only | fair576 | +0.124 | [-0.305; +0.540] |
| A2 fusion không phục hồi trừ A2 full không phục hồi | fair576 | +0.826 | [-0.072; +1.727] |
| A0 full: S1b cũ trừ không phục hồi | native | +1.094 | [+0.275; +1.902] |
| A0 full: S1b mới trừ S1b cũ | native | -0.858 | [-1.425; -0.293] |
| A0 crop: S1b cũ trừ không phục hồi | native | +1.588 | [+0.866; +2.293] |
| A0 crop: S1b mới trừ S1b cũ | native | -0.454 | [-1.072; +0.161] |
| A2 fusion: phục hồi crop qua A2 trừ không phục hồi | native | +0.156 | [-0.162; +0.488] |
| A2 crop: phục hồi qua A2 trừ CE-only | native | +0.408 | [-0.100; +0.919] |
| A2 fusion: phục hồi qua A2 trừ CE-only | native | +0.162 | [-0.172; +0.501] |
| A2 fusion không phục hồi trừ A2 full không phục hồi | native | +2.224 | [+1.615; +2.825] |

## Chi phí một số cấu hình LR4

| Cấu hình | Budget | Token TB | Reader p50 ms | Pipeline p50 ms* | Reader TFLOPs** | Peak CUDA GiB |
|---|---|---:|---:|---:|---:|---:|
| A0 full không phục hồi | fair576 | 576.00 | 166.30 | 166.30 | 6.6982 | 7.432 |
| A0 full + S1b cũ | fair576 | 576.00 | 166.66 | 166.66 | 6.6994 | 7.432 |
| A0 crop + S1b mới | fair576 | 576.00 | 166.07 | 320.92 | 6.6994 | 7.403 |
| A2 full không phục hồi | fair576 | 576.00 | 179.87 | 179.87 | 6.6994 | 7.417 |
| A2 fusion không phục hồi | fair576 | 576.00 | 179.95 | 335.47 | 6.6563 | 7.415 |
| A2 fusion + phục hồi crop qua A2 | fair576 | 576.00 | 180.54 | 335.67 | 6.6569 | 7.435 |
| A0 full không phục hồi | native | 929.42 | 244.89 | 244.89 | 11.6550 | 7.488 |
| A0 full + S1b cũ | native | 929.42 | 244.71 | 244.71 | 11.6570 | 7.488 |
| A0 crop + S1b mới | native | 393.40 | 136.19 | 290.32 | 5.8688 | 7.454 |
| A2 full không phục hồi | native | 929.42 | 263.79 | 263.79 | 11.6569 | 7.467 |
| A2 fusion không phục hồi | native | 1320.99 | 344.37 | 498.40 | 17.5288 | 7.610 |
| A2 fusion + phục hồi crop qua A2 | native | 1320.99 | 344.74 | 498.70 | 17.5298 | 7.615 |

*Timing vision/restorer được tái dùng, selector lấy từ archive, không phải latency end-to-end đo lại. Tổng có selector khiến crop ít token vẫn có thể chậm hơn full. Peak CUDA có các module cố định cùng nằm trong tiến trình.
**FLOPs chỉ gồm các operator PyTorch hỗ trợ, vision/restorer/LLM với 8 token decode cố định, đo trên 3 ảnh. Không gồm selector và operator chưa hỗ trợ; không phải FLOPs toàn pipeline chính xác.

## Kết luận và bước tiếp theo

1. **S1b cũ vẫn là mốc phục hồi đáng giữ.** Với A0 native full, đạt 64,900% so với A0 63,806%, +1,094 điểm [0,275; 1,902]. S1b full mới đạt 64,042%, thấp hơn cũ 0,858 điểm [-1,425; -0,293]. Đây là bằng chứng cho các checkpoint hiện tại, một seed; không suy ra early stopping luôn kém hoặc module mới không thể cải thiện.
2. **Train riêng crop chưa chứng minh tốt hơn dùng S1b cũ train full.** A0 crop mới so cũ: fair +0,086 điểm, native -0,454 điểm; cả hai CI chứa 0. Train dài/crop-specific không tự bảo đảm kết quả tốt hơn.
3. **Train phục hồi qua A2 có tín hiệu ghép tốt hơn, nhưng lợi ích so baseline chưa rõ.** A2 native fusion + module crop đạt 67,512%, so 67,356% không phục hồi: +0,156 [-0,162; 0,489]. Fair 66,040% so 65,960%: +0,080 [-0,293; 0,451]. Chưa vượt CE-only với CI loại 0. Không gọi điểm cao nhất là thắng có ý nghĩa thống kê.
4. **Ghép full + crop có lợi khi cho thêm token; cùng budget còn chưa rõ.** A2 native fusion không phục hồi hơn full +2,224 [1,615; 2,825], nhưng dùng 1.321 thay vì 929 token. Fair +0,826 [-0,072; 1,727]. Cần giữ hai kết luận tách biệt.
5. **CLEAN cần theo dõi khi ghép reader.** S1b full mới qua A2 native làm CLEAN 80,902% xuống 78,990% và LR4 65,132% xuống 63,612%. Module train qua A0 không được chọn trên A2; không quy kết đây là lỗi kiểm tra CLEAN trong training qua A0.
6. Đồ án có thể chốt baseline phục hồi S1b cũ và ghi kết quả mở rộng trung thực. Việc tiếp theo nên là EDA theo cặp cùng câu hỏi: feature gần CLEAN nhưng đáp án sai, trả lời thừa, sai ký tự, và crop có/thiếu vùng đáp án. Không cần tự động train thêm hoặc tìm checkpoint tốt nhất theo 5.000 câu.

## Artifact

Run HPC:
```text
/datastore/dungmt/textvqa/Tuan/qlrg-checkpoint-matrix-3ca5992/runs/checkpoint-matrix-bzv8yp02
```
Bảng gốc và metadata tải local: `local/checkpoint-matrix-112448/` trong repository; gồm `comparison_table.csv`, `paired.csv`, `flops.csv`, `summary.json`, `run.json`, `audit_2026-11-26.json`. Dự đoán đầy đủ và error cases vẫn ở HPC. Không chỉnh sửa/xóa file HPC, không submit job mới trong lần kiểm tra này.
