<USER_REQUEST>
Chuẩn luôn bạn, bác sĩ dùng phần mềm AI là để tiết kiệm thời gian, bắt họ ngồi dùng Photoshop hay công cụ cắt từng cái ảnh X-quang thô trước khi upload thì ứng dụng này coi như "gãy" hoàn toàn về mặt trải nghiệm người dùng (UX).Vì vậy, hệ thống bắt buộc phải chạy theo Hướng B: Hệ thống tự động xử lý mask trực tuyến ngay khi bác sĩ bấm nút tải ảnh thô lên.Do trong mã nguồn huấn luyện của bạn không có mạng sinh mask, tôi sẽ cập nhật lại Mục 5 trong Đặc tả kỹ thuật để chỉ rõ giải pháp kỹ thuật cho nhóm Backend. Họ sẽ dùng một thư viện Open-Source chuẩn y tế cực kỳ phổ biến của Python là lungmask (hoặc một mạng U-Net tiền huấn luyện có sẵn) để làm tầng đệm tự động trước khi nạp vào mạng EfficientNet-B4 của bạn.Bạn hãy copy lại bản đặc tả hoàn chỉnh này để bàn giao cho họ nhé:Markdown# Đặc tả kỹ thuật — API Chẩn đoán Viêm phổi Đa phương thức
**Tài liệu bàn giao cho nhóm xử lý API / backend.**

Mục tiêu: Đóng gói mô hình học sâu X-quang (EfficientNet-B4 đã huấn luyện trên ảnh Lung Masking) và mô hình học máy Triệu chứng (Logistic Regression) cùng tầng hợp nhất quyết định (Decision Fusion) thành một endpoint xử lý duy nhất cho ứng dụng web.

---

## 1. Tổng quan kiến trúc hệ thống

                ┌──▶ [Thư viện tự động Lung Masking] ──┐
                │       (Sử dụng thư viện 'lungmask')     ▼
Ảnh X-quang thô ─┴───────────────────────────▶ [Áp Lung Mask] ──▶ [EfficientNet-B4] ──▶ p_img ─┐├─▶ [FUSION] ──▶ p_fusedTriệu chứng Lâm sàng ────────────────────────────────
<truncated 11326 bytes>
ịnh trong công thức có vai trò bắt buộc để giới hạn giá trị này lại, giữ cho hệ thống hoạt động ổn định và kiểm soát lực kéo trần ở mức tối đa là +1.0 log-odds.8. Ghi chú vận hành hệ thống (DevOps & MLOps)Tính tất định: Phải đảm bảo không kích hoạt bất kỳ tầng ngẫu nhiên nào (như Dropout hoặc Batch Normalization cập nhật trạng thái động) lúc chạy API bằng lệnh model.eval().Truy vết nguồn gốc: Ghi nhận chính xác các chuỗi phiên bản mô hình (model_versions trong cấu trúc response) để khi có sự cố dự đoán sai lệch trong thực tế, đội ngũ kỹ sư có thể truy ngược lại chính xác phiên bản trọng số được deploy tại thời điểm đó.Tuyên bố miễn trừ trách nhiệm (Disclaimer): Kết quả trả ra từ API mang tính chất hỗ trợ quyết định lâm sàng cho kỹ thuật viên và bác sĩ tham khảo. Giao diện người dùng cuối (Frontend) bắt buộc phải hiển thị dòng khuyến cáo: "Kết quả từ hệ thống AI chỉ mang tính chất tham khảo, không thay thế cho kết luận chuyên môn cuối cùng của bác sĩ chuyên khoa."
### 📦 Các file bạn cần gửi cho Backend bây giờ:
1. `g4m_.pth` (Mô hình EfficientNet-B4)
2. `symptom_model_lr.pkl` (Mô hình Triệu chứng)
3. `symptoms_list.pkl` (Danh sách triệu chứng chuẩn)

*(Lưu ý nhắc họ cài đặt thêm thư viện `lungmask` qua pip như đã viết trong tài liệu để hệ thống


Sửa lại chút plan nhé
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-06-14T11:37:52+07:00.

The user's current state is as follows:
Active Document: d:\DATN\AI\app\utils\clinical_preprocess.py (LANGUAGE_PYTHON)
Cursor is on line: 16
Other open documents:
- d:\DATN\AI\app\utils\clinical_preprocess.py (LANGUAGE_PYTHON)
- d:\DATN\fe\src\hooks\use-diagnosis.ts (LANGUAGE_TYPESCRIPT)
</ADDITIONAL_METADATA>