import os
import torch
import logging
from typing import Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """
    Singleton service to manage the loading and execution of the fine-tuned LLM.
    Uses lazy loading to prevent delays during API startup.
    Supports CUDA hardware acceleration and falls back to simulation mode on CPU.
    """
    _instance = None
    _model = None
    _tokenizer = None
    _is_loaded = False
    _is_fallback = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """Loads the tokenizer and base model + LoRA adapter weights."""
        if self._is_loaded:
            return

        if not settings.ENABLE_LLM:
            logger.warning("LLM generation is disabled in configuration settings.")
            self._is_fallback = True
            self._is_loaded = True
            return

        # Check for CUDA availability
        cuda_available = torch.cuda.is_available() and settings.DEVICE == "cuda"
        if not cuda_available:
            logger.warning(
                f"CUDA is not active (device={settings.DEVICE}). "
                "Hugging Face 4-bit quantized models cannot be loaded efficiently on CPU. "
                "LLM Service will run in Simulation (Fallback) Mode."
            )
            self._is_fallback = True
            self._is_loaded = True
            return

        try:
            logger.info("Initializing Hugging Face LLM Service (CUDA)...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            adapter_path = settings.LLM_MODEL_PATH
            base_model_name = settings.LLM_BASE_MODEL

            logger.info(f"Loading tokenizer from: {adapter_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)

            logger.info(f"Configuring 4-bit BitsAndBytes for base model: {base_model_name}")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            logger.info("Loading base model (this might take a minute)...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )

            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            self._model = PeftModel.from_pretrained(base_model, adapter_path)
            self._model.eval()

            self._is_fallback = False
            self._is_loaded = True
            logger.info("LLM model and LoRA adapter loaded successfully on GPU.")

        except Exception as e:
            logger.error(
                f"CRITICAL: Failed to load LLM model locally: {e}. "
                "Exiting to Simulation (Fallback) Mode for stability.",
                exc_info=True
            )
            self._is_fallback = True
            self._is_loaded = True

    def generate_report(self, prompt: str) -> Tuple[str, bool]:
        """
        Generates a professional diagnosis review report from the LLM.
        
        Args:
            prompt: The formatted case details (Master Prompt).
            
        Returns:
            A tuple of (generated_report_string, is_fallback_mode_boolean).
        """
        # Ensure model is initialized (lazy-loaded)
        if not self._is_loaded:
            self.load_model()

        if self._is_fallback or self._model is None:
            logger.info("Generating report using Simulation Mode (CPU Fallback).")
            return self._generate_simulation_report(prompt), True

        try:
            logger.info("Running LLM inference on GPU...")
            
            # Format prompt with Qwen-2.5 template if needed, or send prompt directly
            # Qwen-2.5 Instruct format is typically:
            # <|im_start|>system\nYou are a medical expert...<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
            messages = [
                {"role": "system", "content": "Bạn là một Hội đồng chuyên gia y khoa cấp cao thẩm định chẩn đoán viêm phổi từ hệ thống AI Multimodal."},
                {"role": "user", "content": prompt}
            ]
            
            # Use chat template if available
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.4, # Low temp for medical precision
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Extract only the generated output tokens
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip(), False

        except Exception as e:
            logger.error(f"Error during LLM inference execution: {e}", exc_info=True)
            # Fall back to simulation instead of throwing error to keep the system online
            return (
                f"LƯU Ý: Không thể tạo báo cáo bằng LLM thực tế do lỗi suy luận ({str(e)}).\n\n"
                f"{self._generate_simulation_report(prompt)}"
            ), True
    def _generate_simulation_report(self, prompt: str) -> str:
        """Generates a high-quality simulation report mimicking the fine-tuned LLM output."""
        vision_prob = "N/A"
        clinical_prob = "N/A"
        final_score = "N/A"
        symptoms_str = "Không có"
        curb_str = "Chưa đánh giá"

        # Extract values from the prompt using basic string parsing
        for line in prompt.split("\n"):
            if "Xác suất Vision AI" in line:
                vision_prob = line.split(":")[-1].strip()
            elif "Xác suất Clinical AI" in line:
                clinical_prob = line.split(":")[-1].strip()
            elif "Xác suất Tổng hợp" in line:
                final_score = line.split(":")[-1].strip()
            elif "Triệu chứng khai báo" in line:
                symptoms_str = line.split(":")[-1].strip()
            elif "Thang điểm lâm sàng CURB-65" in line:
                curb_str = line.split(":")[-1].strip()

        # Parse score to numeric for conditional assessment text
        try:
            score_num = float(final_score.replace("%", "").strip()) / 100.0
        except ValueError:
            score_num = 0.0

        # Rule-based diagnostics based on symptoms
        diag_notes = []
        lower_symptoms = symptoms_str.lower()
        if "rusty_sputum" in lower_symptoms:
            diag_notes.append("Đặc biệt ghi nhận triệu chứng đờm màu rỉ sắt, đây là dấu hiệu lâm sàng điển hình chỉ điểm sự hiện diện của phế cầu khuẩn (Streptococcus pneumoniae).")
        if "high_fever" in lower_symptoms and "cough" in lower_symptoms and "phlegm" in lower_symptoms:
            diag_notes.append("Tập hợp triệu chứng sốt cao, ho kèm đờm hướng nhiều đến bệnh cảnh Viêm phổi điển hình (Typical Pneumonia) do vi khuẩn.")
        elif "cough" in lower_symptoms and "fatigue" in lower_symptoms and "high_fever" not in lower_symptoms:
            diag_notes.append("Bệnh cảnh có triệu chứng ho khan, mệt mỏi nhưng không sốt cao, gợi ý khả năng Viêm phổi không điển hình (Atypical Pneumonia) do Mycoplasma hoặc Chlamydia.")
        
        if "breathlessness" in lower_symptoms or "fast_heart_rate" in lower_symptoms:
            diag_notes.append("Có biểu hiện khó thở hoặc nhịp tim nhanh, cần cảnh giác nguy cơ suy hô hấp cấp hoặc biến chứng nhiễm trùng huyết.")
        
        diag_interpretation = " ".join(diag_notes) if diag_notes else "Biểu hiện triệu chứng lâm sàng ở mức độ thông thường, cần theo dõi sát."

        # Action guidelines
        if score_num >= settings.HIGH_RISK_THRESHOLD:
            assessment = "Cảnh báo Nguy cơ Cao. Sự tương quan chặt chẽ giữa hình ảnh tổn thương phổi và triệu chứng lâm sàng cho thấy khả năng viêm phổi đang diễn tiến cấp tính."
            actions = (
                "- **Nhập viện cấp cứu**: Chuyển người bệnh đến cơ sở y tế gần nhất có giường bệnh nội trú.\n"
                "- **Cận lâm sàng khẩn cấp**: Tiến hành đếm công thức máu (WBC, Neutrophil), đo CRP định lượng, và cấy đờm làm kháng sinh đồ.\n"
                "- **Chẩn đoán hình ảnh bổ sung**: Cân nhắc chụp cắt lớp vi tính lồng ngực (CT-Scan) nếu có nghi ngờ tràn dịch màng phổi hoặc áp-xe phổi dạng kén.\n"
                "- **Liệu pháp oxy**: Bắt đầu hỗ trợ thở oxy mask hoặc gọng kính nếu độ bão hòa SpO2 < 94%."
            )
        elif score_num >= settings.MEDIUM_RISK_THRESHOLD:
            assessment = "Nguy cơ Trung bình. Ghi nhận tổn thương nhẹ hoặc không đồng thuận hoàn toàn giữa hình ảnh học và biểu hiện triệu chứng."
            actions = (
                "- **Khám chuyên khoa hô hấp**: Khám lâm sàng nghe phổi để tìm rale ẩm, rale nổ.\n"
                "- **Theo dõi sát tại nhà**: Đo SpO2 và đếm nhịp thở 2 lần/ngày. Yêu cầu nhập viện ngay nếu nhịp thở > 22 lần/phút hoặc SpO2 < 95%.\n"
                "- **Xét nghiệm bổ sung**: Làm xét nghiệm máu ngoại vi và chỉ số viêm (CRP) để quyết định sử dụng kháng sinh ngoại trú."
            )
        else:
            assessment = "Nguy cơ Thấp. Hệ thống chưa phát hiện dấu hiệu viêm phổi rõ rệt từ cả hai phương thức X-quang và Lâm sàng."
            actions = (
                "- **Điều trị triệu chứng tại nhà**: Giảm ho, hạ sốt, uống nhiều nước ấm và nghỉ ngơi hợp lý.\n"
                "- **Theo dõi diễn tiến bệnh**: Tái khám sau 3 ngày hoặc khi có biểu hiện sốt cao không hạ hoặc khó thở tăng lên."
            )

        # Extract weights from prompt
        vision_weight_pct = "70%"
        clinical_weight_pct = "30%"
        for line in prompt.split("\n"):
            if "Vision AI" in line and "Trọng số" in line:
                vision_weight_pct = line.split(":")[-1].strip()
            elif "Clinical AI" in line and "Trọng số" in line:
                clinical_weight_pct = line.split(":")[-1].strip()

        criticism = f"Phân bổ trọng số {vision_weight_pct} Hình ảnh và {clinical_weight_pct} Lâm sàng là phù hợp và an toàn đối với ca bệnh hiện tại."
        if "50%" in clinical_weight_pct:
            criticism += " Cơ chế tăng trọng số lâm sàng lên 50% được kích hoạt tự động do điểm số lâm sàng/CURB-65 thuộc nhóm nguy cấp, giúp nâng cao tính an toàn và giảm thiểu rủi ro âm tính giả từ hình ảnh học."
        else:
            criticism += " Hình ảnh X-quang giữ vai trò chủ đạo để xác định tổn thương thực thể ở nhu mô phổi, tránh bỏ sót các ca viêm phổi ít triệu chứng cơ năng."

        report = f"""## 🏥 HỘI ĐỒNG THẨM ĐỊNH AI MULTIMODAL - BÁO CÁO PHÂN TÍCH LÂM SÀNG

*(Báo cáo mô phỏng do chạy trên CPU không có tăng tốc phần cứng GPU)*

### 1. Phân Tích Sự Đồng Thuận Lâm Sàng & Hình Ảnh:
- **Chỉ số X-quang (Vision):** {vision_prob}
- **Chỉ số Triệu chứng (Clinical):** {clinical_prob}
- **Điểm số Tổng hợp (Final Score):** {final_score}
- **Nhận định chung:** {assessment}

### 2. Biện Giải Hình Ảnh Học & Grad-CAM:
- Vùng nhận diện tổn thương trên phim X-quang ngực thẳng (vùng đỏ/cam trên bản đồ Grad-CAM) tập trung phân tích tại khu vực phế trường. Phù hợp với các dấu hiệu thâm nhiễm phế nang (alveolar infiltration), bóng mờ rải rác hoặc hội tụ đường phế quản.
- **Diễn giải triệu chứng lâm sàng:** {diag_interpretation}

### 3. Khuyến Nghị Lâm Sàng Tiếp Theo:
{actions}

### 4. Phê Bình Tỷ Lệ Trọng Số Phân Bổ:
- {criticism} Tuy nhiên, chẩn đoán cuối cùng phải luôn được cá nhân hóa bởi bác sĩ điều trị dựa trên diễn tiến thực tế của bệnh nhân.
"""
        return report


    def generate_chat_response(self, messages: list) -> Tuple[str, bool]:
        """
        Generates a conversational response from the fine-tuned LLM.
        
        Args:
            messages: A list of dicts with 'role' and 'content' keys.
            
        Returns:
            A tuple of (generated_response_string, is_fallback_mode_boolean).
        """
        # Ensure model is initialized (lazy-loaded)
        if not self._is_loaded:
            self.load_model()

        if self._is_fallback or self._model is None:
            logger.info("Generating chat response using Simulation Mode (CPU Fallback).")
            return self._generate_simulation_chat(messages), True

        try:
            logger.info("Running LLM chat inference on GPU...")
            
            # Check if a system prompt is already present
            has_system = any(msg.get("role") == "system" for msg in messages)
            formatted_messages = list(messages)
            if not has_system:
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": "Bạn là một Bác sĩ AI chuyên khoa Hô hấp. Hãy trả lời câu hỏi của người bệnh bằng tiếng Việt chuẩn y khoa, ngắn gọn, chính xác, không lặp từ và không sử dụng thuật ngữ dịch máy thô sơ."
                })
            
            formatted_prompt = self._tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3, # Low temperature for medical precision
                    top_p=0.85,
                    repetition_penalty=1.2, # Prevent repetition as trained
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Extract only the generated output tokens
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip(), False

        except Exception as e:
            logger.error(f"Error during LLM chat inference execution: {e}", exc_info=True)
            return (
                f"LƯU Ý: Không thể kết nối với mô hình LLM thực tế do lỗi ({str(e)}). Dưới đây là thông tin mô phỏng:\n\n"
                f"{self._generate_simulation_chat(messages)}"
            ), True

    def _generate_simulation_chat(self, messages: list) -> str:
        """Generates a high-quality simulated response for the clinical chatbot when running on CPU."""
        # Get the content of the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").strip()
                break

        if not user_message:
            return "Chào bác sĩ! Tôi là trợ lý AI chuyên khoa hô hấp PlumoX. Tôi có thể hỗ trợ bác sĩ giải đáp thắc mắc gì hôm nay?"

        lower_msg = user_message.lower()

        # Keyword mapping (similar to the frontend's fallback database but centralized here)
        if any(kw in lower_msg for kw in ["phác đồ", "cap", "cộng đồng"]):
            return (
                "### Phác đồ điều trị Viêm phổi mắc phải cộng đồng (CAP) - Bộ Y tế:\n\n"
                "Phân loại mức độ nặng theo thang điểm **CURB-65**:\n"
                "- **CURB-65 = 0-1 (Nhẹ):** Điều trị ngoại trú.\n"
                "  * *Lựa chọn 1:* Amoxicillin (1g x 3 lần/ngày) hoặc Doxycycline (100mg x 2 lần/ngày).\n"
                "  * *Lựa chọn 2 (Nếu nghi ngờ vi khuẩn không điển hình):* Macrolide (Clarithromycin 500mg x 2 lần/ngày hoặc Azithromycin 500mg/ngày).\n"
                "- **CURB-65 = 2 (Trung bình):** Điều trị nội trú ngắn hạn.\n"
                "  * *Phối hợp:* Beta-lactam tiêm truyền (Ceftriaxone 1-2g/ngày hoặc Cefotaxime 1-2g mỗi 8 giờ) **KẾT HỢP** Macrolide uống/tiêm truyền.\n"
                "  * *Hoặc:* Levofloxacin (750mg/ngày) đơn trị liệu.\n"
                "- **CURB-65 ≥ 3 (Nặng):** Nhập viện điều trị tích cực (ICU nếu CURB-65 ≥ 4).\n"
                "  * *Phối hợp:* Beta-lactam tiêm truyền kháng Pseudomonal (Cefepime hoặc Piperacillin/Tazobactam) **KẾT HỢP** Fluoroquinolone hô hấp (Levofloxacin/Moxifloxacin)."
            )
        elif any(kw in lower_msg for kw in ["phân biệt", "điển hình", "không điển hình", "x-quang"]):
            return (
                "### Phân biệt Viêm phổi điển hình và Không điển hình trên X-quang:\n\n"
                "| Đặc điểm | Viêm phổi điển hình (Thùy) | Viêm phổi không điển hình |\n"
                "| :--- | :--- | :--- |\n"
                "| **Hình ảnh X-quang** | Đông đặc thù phổi rõ rệt, ranh giới rõ, có dấu hiệu phế quản phế nang khí (Air bronchogram). | Tổn thương dạng lưới nốt lan tỏa hai bên, tập trung nhiều ở rốn phổi, thâm nhiễm kẽ phổi. |\n"
                "| **Lâm sàng** | Khởi phát cấp tính, sốt cao, rét run, ho đờm mủ, đau ngực màng phổi. | Khởi phát từ từ, sốt nhẹ, ho khan kéo dài, nhức đầu, mệt mỏi toàn thân. |\n"
                "| **Tác nhân thường gặp** | *Streptococcus pneumoniae, Haemophilus influenzae* | *Mycoplasma pneumoniae, Chlamydia pneumoniae, Legionella* |"
            )
        elif any(kw in lower_msg for kw in ["kháng sinh", "liều dùng", "thuốc"]):
            return (
                "### Khuyến cáo kháng sinh ban đầu cho Người lớn (CAP trung bình - CURB-65 = 2):\n\n"
                "1. **Phác đồ phối hợp (Ưu tiên lựa chọn):**\n"
                "   - **Beta-lactam:** Ceftriaxone (1 - 2g tiêm tĩnh mạch/ngày) hoặc Cefotaxime (1 - 2g tiêm tĩnh mạch mỗi 8 giờ).\n"
                "   - **Macrolide phối hợp:** Azithromycin (500mg uống/ngày) hoặc Clarithromycin (500mg uống 2 lần/ngày).\n"
                "2. **Phác đồ đơn trị liệu (Fluoroquinolone hô hấp):**\n"
                "   - Levofloxacin (750mg tiêm tĩnh mạch hoặc uống/ngày).\n"
                "   - Moxifloxacin (400mg tiêm tĩnh mạch hoặc uống/ngày).\n\n"
                "*Lưu ý: Thời gian điều trị tiêu chuẩn thường từ 5 - 7 ngày và bệnh nhân phải hết sốt ít nhất 48 - 72 giờ trước khi ngưng kháng sinh.*"
            )
        elif any(kw in lower_msg for kw in ["curb", "curb65", "curb-65", "thang điểm"]):
            return (
                "### Thang điểm đánh giá độ nặng Viêm phổi CURB-65:\n\n"
                "Mỗi yếu tố tương ứng với **1 điểm**:\n"
                "1. **C**onfusion: Lú lẫn, giảm tỉnh táo (AMTS ≤ 8).\n"
                "2. **U**rea: Urê huyết > 7 mmol/L (~19 mg/dL).\n"
                "3. **R**espiratory Rate: Nhịp thở ≥ 30 lần/phút.\n"
                "4. **B**lood Pressure: Huyết áp tâm thu < 90 mmHg hoặc huyết áp tâm trương ≤ 60.\n"
                "5. **65**: Tuổi bệnh nhân từ 65 trở lên.\n\n"
                "**Định hướng xử trí lâm sàng:**\n"
                "*   **0 - 1 điểm**: Nguy cơ tử vong thấp (1.5%). Điều trị ngoại trú.\n"
                "*   **2 điểm**: Nguy cơ tử vong trung bình (9.2%). Nhập viện điều trị nội trú ngắn hạn hoặc theo dõi sát.\n"
                "*   **3 - 5 điểm**: Nguy cơ tử vong cao (22% - 57%). Nhập viện điều trị nội trú tích cực (Cân nhắc ICU nếu từ 4 điểm)."
            )
        else:
            return (
                f"Cảm ơn bác sĩ đã chia sẻ câu hỏi về: *\"{user_message}\"*.\n\n"
                "Với vai trò là **Bác sĩ AI chuyên khoa Hô hấp**, để đưa ra hỗ trợ tư vấn lâm sàng chính xác nhất cho ca bệnh viêm phổi này, tôi khuyến nghị bác sĩ cung cấp thêm các thông tin:\n"
                "1. **Hình ảnh học (X-quang ngực)**: Có xuất hiện đám mờ đông đặc thù, tổn thương thâm nhiễm phế nang hay bóng mờ phế quản phế nang khí không?\n"
                "2. **Các triệu chứng cơ năng & dấu hiệu sinh tồn**: Bệnh nhân có bị lú lẫn, nhịp thở (lần/phút), huyết áp (tâm thu/tâm trương) và nồng độ Urê trong máu thế nào (để tính điểm độ nặng CURB-65)?\n"
                "3. **Tính chất ho & đờm**: Ho khan hay ho có đờm (đờm mủ, đờm màu rỉ sắt)?\n\n"
                "*Bác sĩ cũng có thể đặt các câu hỏi trực tiếp về phác đồ kháng sinh CAP của Bộ Y tế, cách phân biệt viêm phổi điển hình/không điển hình hoặc thang điểm đánh giá độ nặng.*"
            )

llm_service = LLMService()
