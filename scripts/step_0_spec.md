<USER_REQUEST>
Đọc toàn bộ đặc tả về các model mới tôi train

Sửa lại logic API của AI nhé...
Markdown# Đặc tả kỹ thuật — API Chẩn đoán Viêm phổi Đa phương thức
**Tài liệu bàn giao cho nhóm xử lý API / backend.**

Mục tiêu: Đóng gói mô hình học sâu X-quang (EfficientNet-B4 đã huấn luyện trên ảnh Lung Masking) và mô hình học máy Triệu chứng (Logistic Regression) cùng tầng hợp nhất quyết định (Decision Fusion) thành một endpoint xử lý duy nhất cho ứng dụng web.

---

## 1. Tổng quan kiến trúc hệ thống

                ┌──▶ [Mô hình Lung Segmentation] ──┐
                │                                    ▼
Ảnh X-quang thô ─┴───────────────────────────▶ [Áp Lung Mask] ──▶ [EfficientNet-B4] ──▶ p_img ─┐├─▶ [FUSION] ──▶ p_fusedTriệu chứng Lâm sàng ───────────────────────────────────────────────▶ [Logistic Regression] ──▶ p_sym ┘
### Nguyên tắc cốt lõi của tầng Fusion
1. **Nhánh X-quang là trụ quyết định chính**: Mô hình ảnh được tối ưu và chọn ngưỡng tại điểm vận hành có độ nhạy cao (0.85). 
2. **Nhánh triệu chứng đóng vai trò bổ trợ lâm sàng (Corroborate)**: Tín hiệu lâm sàng chỉ có tác dụng **nâng** mức độ nghi ngờ ở vùng ranh giới (nudge >= 0), **không hạ** xác suất ảnh và **không lật** được kết luận của ảnh khi ảnh có độ tin cậy cao. Ảnh đóng vai trò là "Sàn quyết định" để bảo toàn độ nhạy tối đa, tránh bỏ sót bệnh.

---

## 2. Hợp đồng API (API Contract)

### 2.1. Request — `POST /api/v1/diagnose`
* **Content-Type**: `multipart/form-data`

| Trường | Kiểu dữ liệu | Bắt buộc | Mô tả 
<truncated 10761 bytes>
nhận chính xác các chuỗi phiên bản mô hình (model_versions trong cấu trúc response) để khi có sự cố dự đoán sai lệch trong thực tế, đội ngũ kỹ sư có thể truy ngược lại chính xác phiên bản trọng số được deploy tại thời điểm đó.Tuyên bố miễn trừ trách nhiệm (Disclaimer): Kết quả trả ra từ API mang tính chất hỗ trợ quyết định lâm sàng cho kỹ thuật viên và bác sĩ tham khảo. Giao diện người dùng cuối (Frontend) bắt buộc phải hiển thị dòng khuyến cáo: "Kết quả từ hệ thống AI chỉ mang tính chất tham khảo, không thay thế cho kết luận chuyên môn cuối cùng của bác sĩ chuyên khoa."

Model thì tôi đã thêm vào rồi đó...
@[d:\DATN\AI\app\models\vision\g4m.pth] 
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-06-14T11:30:45+07:00.

The user's current state is as follows:
Active Document: d:\DATN\fe\src\hooks\use-diagnosis.ts (LANGUAGE_TYPESCRIPT)
Cursor is on line: 1
Other open documents:
- d:\DATN\be\src\main\java\com\medical\pneumonia\dto\response\RoleResponse.java (LANGUAGE_JAVA)
- d:\DATN\AI\app\services\inference_service.py (LANGUAGE_PYTHON)
- d:\DATN\AI\streamlit_app.py (LANGUAGE_PYTHON)
- d:\DATN\be\src\main\java\com\medical\pneumonia\repository\RoleRepository.java (LANGUAGE_JAVA)
- d:\DATN\BE\src\main\java\com\medical\pneumonia\service\UserService.java (LANGUAGE_JAVA)

The user has mentioned some items in the form @[ITEM]. Here is extra information about the items that were mentioned by the user, in the order that they appear:

@[d:\DATN\AI\app\models\vision\g4m.pth] is a [File]:
d:\DATN\AI\app\models\vision\g4m.pth
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Gemini 3.5 Flash (High). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>