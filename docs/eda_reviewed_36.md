# EDA trực quan 36 câu, job 52963

Ngày kiểm tra 19/11/2026. Người thực hiện là trợ lý, đối chiếu ảnh CLEAN/LR4,
câu hỏi, 10 đáp án tham chiếu và dự đoán A0/S1b/Pre/A2. Đây là quan sát lỗi
đầu ra. Các giả thuyết về cơ chế nội bộ cần phép kiểm tra riêng.

Chọn có chủ đích sáu ca mỗi nhóm, loại trùng question ID. Ba nhóm đầu có
khác biệt 1 ↔ 0 rõ ràng; nhóm cuối kiểm tra tác động của điểm một phần.
Số ca của các nhóm được cố định khi chọn, nên không dùng tỷ lệ trong bảng
để ước lượng phân bố lỗi của cả 5.000 câu. Sáu ca đã kiểm tra trước được giữ
để thuận tiện đối chiếu với bản EDA ban đầu.

Ảnh đối chiếu và dữ liệu của từng ca nằm trong thư mục artifact
`manual_eda_36/` của job 52963. Các quan sát dưới đây dựa trên sáu bảng ảnh,
mỗi bảng chứa sáu cặp CLEAN/LR4.

## S1b đúng, Pre đạt 0

| Câu | Quan sát sau khi xem ảnh | Loại khác biệt |
| --- | --- | --- |
| 37947 | Biển số ghi R275 ULO. S1b LR4 trả r275ulo, Pre trả r275 ul0. Có khác ký tự cuối và cách tách chuỗi. Các nhánh vẫn nhầm trên CLEAN. | Ký tự và định dạng |
| 36702 | Câu hỏi nhắm phím góc dưới phải. Pre LR4 trả esc, S1b trả enter. Cả hai trả enter trên CLEAN; chữ phím nhỏ hơn và mờ hơn trên LR4. | Chọn mục tiêu/đọc chữ nhỏ |
| 37614 | Biển có dòng DRIVE-IN phía trên PIZZA PALACE. Pre thêm dòng DRIVE-IN vào tên, điểm 0. S1b giữ tên khớp tham chiếu. | Phạm vi đáp án |
| 36435 | Hộp đồ chơi có các con vật trang trại. Pre LR4 đổi từ farm trên CLEAN sang bear family. S1b giữ farm animals. | Nhận diện hoặc chọn nội dung sai |
| 35601 | Ảnh là sân băng. Pre ice hockey rink có nghĩa phù hợp với cảnh, nhưng bộ nhãn có unanswerable, elmira college và ice rink. S1b unanswerable được 1. | Mơ hồ nhãn và cách diễn đạt |
| 39196 | Bảng nhỏ thể hiện giá $5. Pre LR4 viết $5.00, S1b $5. Cùng giá trị số nhưng scorer chỉ cho $5 điểm đầy đủ. | Định dạng số |

## Pre đúng, S1b đạt 0

| Câu | Quan sát sau khi xem ảnh | Loại khác biệt |
| --- | --- | --- |
| 37183 | Bao bì ghi GOLDEN DAWN. S1b LR4 đọc pawn, Pre dawn. CLEAN cả hai đúng. | Ký tự D/P |
| 35144 | Nhãn giữa ảnh ghi TOCCA. S1b thêm s trên cả CLEAN và LR4. Pre giữ TOCCA. | Thêm ký tự, xuất hiện cả CLEAN |
| 36855 | APEROL là chữ lớn, BARBIERI nằm nhỏ phía dưới. S1b LR4 trả aperol, Pre trả đúng chữ ở vị trí câu hỏi. | Chọn đúng dòng chữ theo câu hỏi |
| 38546 | Tên Renaud Camus ở đầu bìa sách, Fayard ở cuối. S1b CLEAN chọn nhà xuất bản; LR4 viết reinaud camus. Pre giữ đúng tên. | Chọn trường thông tin và ký tự |
| 38484 | Nhãn ghi Atomic Pale Ale. Pre LR4 trả atomic, S1b thêm pale ale. Những từ thêm có thật trên ảnh. | Phạm vi đáp án |
| 36057 | Bảng Via 1 ghi RECOL-F.MORA. S1b LR4 bỏ phần F; Pre giữ phần đó. Cách chấm còn phụ thuộc chuẩn hóa dấu và khoảng trắng. | Thiếu thành phần chuỗi |

## S1b khôi phục câu A0 đạt 0

| Câu | Quan sát sau khi xem ảnh | Loại khác biệt |
| --- | --- | --- |
| 37299 | Có biển Bridgestone dọc trên tòa nhà. A0 LR4 trả no; S1b/Pre/A2 trả yes. CLEAN các nhánh trả yes. | Nhận biết sự hiện diện chữ nhỏ |
| 39190 | Gian hàng có nhiều bảng nhỏ. A0 LR4 no chuyển thành yes ở các nhánh học thêm. Thumbnail chưa đủ đọc chắc vị trí logo Visa. | Cải thiện điểm; bằng chứng chữ cần zoom thêm |
| 37570 | Ảnh có dòng THE ORNAMENT dưới DAY 66. A0 LR4 viết THE JAMMINT; các nhánh học thêm đọc đúng. | Đọc chữ nhỏ trên nền tối |
| 39254 | Phông nền ghi OSCARS. A0 dùng oscar cả CLEAN/LR4; S1b dùng oscars. | Hình thái từ/chuỗi tham chiếu |
| 35089 | Áo đội bóng có TEXAS. A0 LR4 viết texas rangers; S1b chỉ trả texas. | Phạm vi tên đội |
| 36309 | Ảnh có chữ Peace, câu hỏi “this peaceful place?” và nhiều loại đáp án. S1b yes được 1, A0 peace được 0. | Câu hỏi mơ hồ và loại đáp án |

## S1b làm hỏng câu A0 đạt 1

| Câu | Quan sát sau khi xem ảnh | Loại khác biệt |
| --- | --- | --- |
| 35207 | Ảnh có banner MONTRÉAL và banner New York City. A0 LR4 Montreal đúng, các nhánh học thêm montral thiếu e. CLEAN lại chọn New York City, được điểm một phần. | Ký tự và nhiều vùng chữ |
| 35575 | Nhãn ghi WEDDING ALE. S1b trả beer, nghĩa rộng hơn đáp án. Lỗi đã có trên CLEAN. | Mức chi tiết của đáp án |
| 37598 | Trên xe buýt có Liverpool. A0 giữ Liverpool, các nhánh học thêm LR4 đổi thành London; CLEAN đều đúng. | Tên riêng/địa danh sai |
| 35417 | Sản phẩm ghi facial treatment; câu hỏi hỏi bộ phận cơ thể. A0 face được 1, S1b facial được 0 trên cả hai điều kiện. | Dạng từ và loại câu trả lời |
| 36411 | Biển có cả Do not enter và Demolition work in progress. Câu hỏi hỏi vì sao nguy hiểm. S1b/Pre chọn lời cấm, A0/A2 chọn lý do. | Chọn dòng chữ và quan hệ với câu hỏi |
| 38141 | Câu hỏi dẫn “yes, the sale sign is here?” trong ảnh màn hình TV. A0 no được 1, các nhánh học thêm yes được 0. Nhãn có nhiều “not a question”. | Câu hỏi dẫn và chất lượng nhãn |

## Cả hai restorer đạt 0 trên LR4

| Câu | Quan sát sau khi xem ảnh | Loại khác biệt |
| --- | --- | --- |
| 38361 | Nhà phía sau ghi BECKER AUTO BODY. Hai restorer LR4 viết recker. CLEAN becker được 0,6. Câu hỏi “where” còn có nhãn Illinois và unanswerable. | Ký tự và mơ hồ nhãn |
| 38630 | Biển số nhỏ, nghiêng trên xe vàng. Tất cả nhánh đã đọc sai cả CLEAN; LR4 có thêm lỗi ký tự/số. | Chuỗi nhỏ, nghiêng; giới hạn đã có ở CLEAN |
| 36045 | Danh sách đầu cột có SAIGON, phía dưới nhiều địa danh. CLEAN các nhánh đúng; LR4 S1b/Pre bang dinh. | Chữ nhỏ và thứ tự dòng |
| 38831 | Chú thích ảnh cũ là chữ viết tay nhỏ. CLEAN các nhánh đọc được Henry D. Morse; LR4 đổi phần họ thành man. | Chữ viết tay bị suy giảm |
| 36178 | Hai icon xanh LINE và Hangouts nằm trong cùng ảnh. S1b/Pre LR4 chọn Hangouts thay LINE. A2 giữ đúng LINE. | Nhầm mục tiêu giữa các icon/chữ gần nhau |
| 39191 | Câu hỏi yêu cầu liệt kê các thẻ được chấp nhận. CLEAN chỉ trả Visa and Discover, thiếu phần danh sách theo nhãn; LR4 càng thiếu. Các nhãn cũng có nhiều cách viết. | Thiếu danh sách và khác dạng chuỗi |

## Khác biệt điểm một phần

| Câu | Quan sát sau khi xem ảnh | Loại khác biệt |
| --- | --- | --- |
| 38310 | Xe có dòng National Severe Storms Laboratory. Pre LR4 trả weather theo cảnh bão; S1b unanswerable được 0,3 vì có một nhãn như vậy. | Ưu tiên nội dung cảnh và điểm từ nhãn thiểu số |
| 38243 | Câu hỏi “could you tell me the time?” chủ yếu có nhãn giờ. S1b yes được 0,3 vì một người ghi yes. Điểm đó chưa thể hiện đọc giờ đúng. | Câu hỏi mơ hồ, điểm một phần |
| 36690 | Xe ghi J.V.A. SUNSEEKER TOURS. Câu hỏi hỏi loại xe, nhãn lẫn loại và tên hãng, có lỗi chính tả. CLEAN tour bus được 0 ở các nhánh. | Mơ hồ loại thông tin và nhãn |
| 34959 | Biển nhỏ phía trái ghi Green St. Pre LR4 đọc green được 0,6, S1b alt được 0; CLEAN đều Green St được 1. | Đọc thiếu thành phần/nhầm biển gần nhau |
| 36768 | Poster ghi cả Floyd Bennett Field và North Beach. Pre LR4 giữ cả hai được 0,6; S1b chỉ giữ một nơi được 0. | Độ đầy đủ câu trả lời và nhiều cách diễn đạt |
| 38247 | Ảnh là lon Mtn Dew Baja Blast. Pre LR4 trả beer sai loại đồ uống. S1b Baja Blast được 0,3 do số người ghi biến thể này ít. | Sai loại đồ uống và điểm theo đồng thuận |

## Điều rút ra để viết kết luận

- Có ca được cải thiện về ký tự và vùng chữ, như 37183 và 36855.
- Có ca thay đổi cách trả lời trên cả CLEAN và LR4, như 35144, 39254 và 35417.
  Những ca này cần được đọc cùng đối chứng thích nghi A2.
- Có ca đọc thêm chữ có thật trên ảnh nhưng bị chấm 0 do phạm vi đáp án,
  như 37614 và 38484. Cần xem ảnh trước khi gán nhãn ảo giác.
- Câu hỏi và đáp án tham chiếu đôi khi mơ hồ. Soft accuracy phản ánh đồng
  thuận giữa 10 người, chưa mô tả đầy đủ mọi cách diễn đạt hợp nghĩa.
- Tất cả nhánh nhận toàn ảnh trong phép so này. Các lỗi chọn dòng và ngữ
  cảnh ở đây xuất hiện trong model nhận toàn ảnh.
- Cần thí nghiệm can thiệp mới để quy nguyên nhân cho nét chữ được khôi
  phục, mức chú ý hay ngôn ngữ. EDA này mô tả những khác biệt quan sát được.
