# Runbook demo và kiểm tra kết quả Q-LRG

## Trạng thái ngày 19/11/2026

- Job 48633 hoàn tất train Pre và chấm val_inner.
- Job 52963 hoàn tất chấm bốn checkpoint trên cùng 5.000 câu CLEAN/LR4.
- Checksum 16 artifact của 52963 và accuracy tính lại từ predictions đã qua kiểm tra.
- Trang phát lại có ảnh đã có trong `demo.html`.
- Job 63914 kiểm chứng CLI suy luận với ba ảnh đã submit. Lần kiểm tra gần
  nhất là PENDING, lý do AssocGrpJobsLimit. Demo CLI trên H200 còn chờ kết quả này.

## Vị trí artifact

```bash
main=/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA
crop=/datastore/dungmt/textvqa/Tuan/Q-LRG-crop
evaluation="$crop/runs/final-tooling-xszFH1L0/runs/final-official_val-5zxrc8y1"
demo_check="$crop/runs/final-demo-check-WI4KmnsB"
```

Tệp chính gồm `accuracy.csv`, `paired.csv`, `degradation_interaction.csv`,
`cost_summary.csv`, `error_cases.csv`, `review_lr4.html`, `review_clean.html`
và `demo.html`. Các HTML là tệp mở bằng trình duyệt sau khi tải về.

## Kiểm tra tính toàn vẹn

Từ checkout có script kiểm tra mới

```bash
python3 scripts/verify_final_artifacts.py "$evaluation"
```

Kết quả mong đợi là xác thực 5.000 câu, 3.166 ảnh, bốn cấu hình, hai điều
kiện; checksum các artifact và accuracy tính lại đều khớp. Script chỉ dùng
Python standard library, chạy được trên login node hoặc trên gói tải về.

## Kiểm chứng CLI trên H200

Job 63914 gọi `run_final_demo.py` với đường dẫn ảnh và câu hỏi truyền vào,
lần lượt ba ảnh của câu 37947, 37183 và 36855. Mỗi lần load các checkpoint
đã khóa và phải sinh đủ tám kết quả, bốn cấu hình nhân CLEAN/LR4.
Các ảnh được chọn để kiểm tra chức năng CLI, đã nằm trong tập benchmark.

```bash
sacct -j 63914 --format=JobID,JobName,State,ExitCode,Elapsed
tail -F "$demo_check/logs/qlrg-demo-check-63914.out" \
        "$demo_check/logs/qlrg-demo-check-63914.err"
```

Khi job thành công, có `37947.json`, `37183.json`, `36855.json` và
`COMPLETE.json` trong thư mục demo_check. Log phải có
`Live inference demo passed on 3 images`.

Trong allocation GPU đang có, thực hiện demo với ảnh tùy chọn

```bash
cd "$evaluation/code"
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
"$main/.venv/bin/python" -B run_final_demo.py \
  --evaluation "$evaluation" \
  --image /absolute/path/to/image.jpg \
  --question "What does the sign say?" \
  --output /absolute/path/to/new-demo-result.json
```

Đầu ra có nhãn cấu hình, đáp án, token và thời gian generation. Lệnh này
không sử dụng đáp án tham chiếu của ảnh được nạp. Nếu ảnh đầu vào đã mờ,
nhãn CLEAN trong CLI chỉ biểu thị chính ảnh nạp vào, còn LR4 là ảnh đó bị
giảm thêm 4× rồi phóng lại. Khi trình bày cần gọi là input và input-LR4
để tránh hiểu nhầm chất lượng ảnh gốc.

## Luồng demo phục vụ báo cáo

1. Nêu ảnh và câu hỏi, nhìn thấy chữ cần đọc trong ảnh input.
2. Cho xem LR4 cùng kích thước và giải thích cách tạo suy giảm.
3. Cho xem A0 và S1b, sau đó bật Pre/A2 để đối chiếu.
4. Nếu có nhãn TextVQA, giải thích điểm từng câu và mười đáp án tham chiếu.
5. Kèm ít nhất một ca bị hỏng sau phục hồi và một ca điểm khác do định dạng.
6. Phân biệt kết quả suy luận vừa sinh với bản phát lại đã lưu.

## Câu hỏi cần nắm để bảo vệ

| Câu hỏi | Căn cứ trả lời |
| --- | --- |
| Model phục hồi gì? | S1b/Pre chỉnh vector visual feature. Ảnh đầu vào được xử lý theo protocol CLEAN/LR4. |
| Vì sao giữ kích thước ảnh sau LR4? | Để giữ cùng lưới và số token cho việc đối chiếu CLEAN/LR4. |
| Teacher CLEAN có dùng khi suy luận? | Teacher và đáp án cung cấp loss lúc train. Demo nhận ảnh và câu hỏi. |
| A2 khác S1b thế nào? | A2 học LoRA Q/V của LLM từ đáp án; S1b học module visual từ feature CLEAN và đáp án. |
| Pre và Post có fair? | Cùng manifest, hai epoch, seed, optimizer và loss sau Merger; tham số chênh 0,16%, FLOPs được báo riêng. |
| Vì sao ba người đồng ý cho 0,9? | Scorer bỏ từng người rồi trung bình mười lượt; ba người khớp cho trung bình 0,9. |
| CI theo ảnh giải quyết gì? | Giữ các câu cùng ảnh trong một cụm khi resample. Cần nhiều seed để đo thêm biến thiên train. |
| S1b có thắng A2? | LR4 S1b 64,958%, A2 65,274%; CI chênh lệch còn qua 0. |
| Đã chứng minh restoration khác adaptation? | S1b tăng cả CLEAN/LR4; interaction CI qua 0, nên kết luận cơ chế còn giới hạn. |
| Có thể khẳng định trước Merger luôn kém? | Bằng chứng hiện tại áp dụng cho module residual, hai epoch và một seed đã thử. |
| Model thêm chữ có phải ảo giác? | Cần xem ảnh. DRIVE-IN có thật trong ca 37614 nhưng nằm ngoài phạm vi nhãn tên nhà hàng. |
| Đã có unseen test chưa? | Bảng chính là validation đã dùng trong dự án; test chính thức hoặc holdout mới là hướng xác nhận thêm. |
