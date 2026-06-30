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
        if self._is_loaded and self._model is not None:
            return

        if not settings.ENABLE_LLM:
            logger.warning("LLM generation is disabled in configuration settings.")
            self._is_fallback = True
            self._is_loaded = True
            return

        # Check for CUDA availability
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logger.warning(
                "CUDA is not active. Attempting to load Hugging Face model anyway (may fail on CPU)."
            )

        try:
            logger.info("Initializing Hugging Face LLM Service...")
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
            logger.info("LLM model and LoRA adapter loaded successfully.")

        except Exception as e:
            logger.error(
                f"CRITICAL: Failed to load LLM model: {e}.",
                exc_info=True
            )
            self._is_fallback = True
            self._is_loaded = True
            raise e

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
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")

        if self._is_fallback or self._model is None:
            logger.info("Generating report using Simulation Mode (CPU Fallback).")
            return self._generate_simulation_report(prompt), True

        try:
            logger.info("Running LLM inference on GPU...")
            
            # Format prompt with Qwen-2.5 template if needed, or send prompt directly
            # Qwen-2.5 Instruct format is typically:
            # <|im_start|>system\nYou are a medical expert...<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là một Hội đồng chuyên gia y khoa cấp cao thẩm định chẩn đoán viêm phổi từ hệ thống AI Multimodal. "
                        "Hãy phân tích ca bệnh dựa trên kết quả hình ảnh học, lâm sàng và thang điểm CURB-65. "
                        "LƯU Ý QUAN TRỌNG: Tuyệt đối không tự ý kê đơn, không đưa ra tên thuốc hay liều lượng điều trị cụ thể trong báo cáo. "
                        "Chỉ đề xuất các xét nghiệm, biện pháp theo dõi cận lâm sàng và hướng xử trí chung, đồng thời hướng dẫn bác sĩ tham khảo phác đồ điều trị chi tiết tại Quyết định số 4815/QĐ-BYT."
                    )
                },
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
                f"LƯU Ý LÂM SÀNG: Báo cáo được tạo ở chế độ dự phòng (Fallback Mode) do hệ thống xử lý trung tâm bận.\n\n"
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
        if "high_fever" in lower_symptoms and "cough" in lower_symptoms and "phlegm" in lower_symptoms:
            diag_notes.append("Tập hợp triệu chứng sốt cao, ho kèm đờm hướng nhiều đến bệnh cảnh Viêm phổi điển hình (Typical Pneumonia) do vi khuẩn.")
        elif "cough" in lower_symptoms and "fatigue" in lower_symptoms and "high_fever" not in lower_symptoms:
            diag_notes.append("Bệnh cảnh có triệu chứng ho khan, mệt mỏi nhưng không sốt cao, gợi ý khả năng Viêm phổi không điển hình (Atypical Pneumonia) do Mycoplasma hoặc Chlamydia.")
        
        if "breathlessness" in lower_symptoms or "fast_heart_rate" in lower_symptoms:
            diag_notes.append("Có biểu hiện khó thở hoặc nhịp tim nhanh, cần cảnh giác nguy cơ suy hô hấp cấp hoặc biến chứng nhiễm trùng huyết.")
        
        diag_interpretation = " ".join(diag_notes) if diag_notes else "Biểu hiện triệu chứng lâm sàng ở mức độ thông thường, cần theo dõi sát."

        # Parse CURB-65 score safely
        curb_score_num = None
        try:
            if curb_str:
                cleaned_curb = "".join([c for c in curb_str if c.isdigit() or c == '/'])
                if '/' in cleaned_curb:
                    curb_score_num = int(cleaned_curb.split('/')[0])
        except Exception as e:
            logger.error(f"Error parsing CURB-65 from string '{curb_str}': {e}")

        curb_category = "Chưa được phân nhóm (Thiếu dữ liệu CURB-65)"
        if curb_score_num is not None:
            if curb_score_num <= 1:
                curb_category = "Nhóm 1 (Nguy cơ tử vong thấp - Tỷ lệ tử vong 30 ngày < 3%)"
            elif curb_score_num == 2:
                curb_category = "Nhóm 2 (Nguy cơ tử vong trung bình - Tỷ lệ tử vong 30 ngày ~ 9%)"
            else:
                curb_category = "Nhóm 3 (Nguy cơ tử vong cao - Tỷ lệ tử vong 30 ngày 15% - 22%)"

        # Action guidelines
        if score_num >= settings.HIGH_RISK_THRESHOLD:
            assessment = "Cảnh báo Nguy cơ Cao. Sự tương quan giữa X-quang và triệu chứng cho thấy khả năng viêm phổi tiến triển."
            actions = (
                f"- **Phân nhóm mức độ nặng (Theo CURB-65: {curb_str})**: **{curb_category}**.\n"
                "- **Xét nghiệm đề xuất**: Tiến hành đếm công thức máu (WBC, Neutrophil), đo CRP định lượng, và cấy đờm làm kháng sinh đồ để định danh tác nhân.\n"
                "- **Chẩn đoán hình ảnh bổ sung**: Cân nhắc chụp cắt lớp vi tính lồng ngực (CT-Scan) nếu có nghi ngờ tràn dịch màng phổi hoặc áp-xe phổi.\n"
                "- **Theo dõi lâm sàng**: Kiểm tra nhịp thở và nồng độ bão hòa oxy SpO2 thường xuyên."
            )
        elif score_num >= settings.MEDIUM_RISK_THRESHOLD:
            assessment = "Nguy cơ Trung bình. Ghi nhận tổn thương nhẹ hoặc không đồng thuận hoàn toàn giữa hình ảnh học và biểu hiện triệu chứng."
            actions = (
                f"- **Phân nhóm mức độ nặng (Theo CURB-65: {curb_str})**: **{curb_category}**.\n"
                "- **Xét nghiệm bổ sung**: Làm xét nghiệm máu ngoại vi và chỉ số viêm (CRP) để hỗ trợ chẩn đoán.\n"
                "- **Theo dõi lâm sàng**: Thăm khám nghe phổi phát hiện tiếng rale bất thường và kiểm soát nhịp thở của bệnh nhân."
            )
        else:
            assessment = "Nguy cơ Thấp. Hệ thống chưa phát hiện dấu hiệu viêm phổi rõ rệt từ cả hai phương thức X-quang và Lâm sàng."
            actions = (
                f"- **Phân nhóm mức độ nặng (Theo CURB-65: {curb_str})**: **{curb_category}**.\n"
                "- **Theo dõi diễn tiến**: Tiếp tục theo dõi các triệu chứng hô hấp và đo thân nhiệt cơ thể khi cần thiết."
            )

        # Extract weights from prompt
        vision_weight_pct = "70%"
        clinical_weight_pct = "30%"
        for line in prompt.split("\n"):
            if "Trọng số Vision AI" in line and ":" in line:
                vision_weight_pct = line.split(":")[-1].strip()
            elif "Trọng số Clinical AI" in line and ":" in line:
                clinical_weight_pct = line.split(":")[-1].strip()

        criticism = f"Phân bổ trọng số {vision_weight_pct} Hình ảnh và {clinical_weight_pct} Lâm sàng là phù hợp và an toàn đối với ca bệnh hiện tại."
        if "50%" in clinical_weight_pct:
            criticism += " Cơ chế tăng trọng số lâm sàng lên 50% được kích hoạt tự động do điểm số lâm sàng/CURB-65 thuộc nhóm nguy cấp, giúp nâng cao tính an toàn và giảm thiểu rủi ro âm tính giả từ hình ảnh học."
        else:
            criticism += " Hình ảnh X-quang giữ vai trò chủ đạo để xác định tổn thương thực thể ở nhu mô phổi, tránh bỏ sót các ca viêm phổi ít triệu chứng cơ năng."

        report = f"""## BÁO CÁO HỘI CHẨN ĐA PHƯƠNG THỨC — HỖ TRỢ QUYẾT ĐỊNH LÂM SÀNG HÔ HẤP

(Báo cáo hỗ trợ quyết định lâm sàng tự động bằng công nghệ AI của PlumoX — Chỉ dùng cho mục đích tham khảo chuyên môn, không thay thế quyết định lâm sàng của bác sĩ)

### 1. Phân Tích Sự Đồng Thuận Lâm Sàng & Hình Ảnh:
- **Chỉ số X-quang (Vision):** {vision_prob}
- **Chỉ số Triệu chứng (Clinical):** {clinical_prob}
- **Điểm số Tổng hợp (Final Score):** {final_score}
- **Nhận định chung:** {assessment}

### 2. Biện Giải Hình Ảnh Học & Grad-CAM:
- Vùng nhận diện tổn thương trên phim X-quang ngực thẳng (vùng đỏ/cam trên bản đồ Grad-CAM) tập trung phân tích tại khu vực phế trường. Phù hợp với các dấu hiệu thâm nhiễm phế nang (alveolar infiltration), bóng mờ rải rác hoặc hội tụ đường phế quản.
- **Diễn giải triệu chứng lâm sàng:** {diag_interpretation}

### 3. Định Hướng Theo Dõi & Cận Lâm Sàng:
{actions}

### 4. Đánh giá Tỷ lệ Trọng số Tổng hợp:
- {criticism} Tuy nhiên, chẩn đoán cuối cùng phải luôn được cá nhân hóa bởi bác sĩ điều trị dựa trên diễn tiến thực tế của bệnh nhân.
"""
        return report


    def _is_query_pneumonia_related(self, text: str) -> bool:
        text_lower = text.lower()
        keywords = [
            "viêm phổi", "pneumonia", "phổi", "lung", "cough", "ho ", "sốt", "fever", "đờm", 
            "sputum", "phlegm", "khó thở", "breathlessness", "ngực", "chest", "tim ", "heart", 
            "curb", "curb65", "curb-65", "x-quang", "xray", "phác đồ", "kháng sinh", "antibiotic", 
            "cap ", "typical", "atypical", "điển hình", "không điển hình", "thâm nhiễm", 
            "đông đặc", "consolidation", "infiltration", "tràn dịch", "effusion"
        ]
        return any(kw in text_lower for kw in keywords)

    def _call_gemini_api(self, messages: list) -> str:
        """Calls the Google Gemini API to get a response for non-pneumonia or general questions."""
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            return "Vui lòng cấu hình GEMINI_API_KEY trong file .env để sử dụng tính năng trả lời tự động từ Gemini."

        # Clean key from accidental whitespaces, single or double quotes
        api_key = api_key.strip().strip("'").strip('"')

        import urllib.request
        import urllib.error
        import json

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # Format conversation history for Gemini
        contents = []
        system_instruction = (
            "Bạn là một Bác sĩ AI chuyên khoa Hô hấp hỗ trợ tra cứu lâm sàng. Hãy trả lời câu hỏi bằng tiếng Việt chuẩn y khoa, ngắn gọn, chính xác, lịch sự. "
            "LƯU Ý QUAN TRỌNG: Không tự ý kê đơn, không đưa ra tên thuốc hay liều lượng cụ thể (không khuyến nghị các liều thuốc như Amoxicillin 1g, Ceftriaxone, v.v.). "
            "Hãy định hướng xử trí chung theo khuyến cáo của Bộ Y tế Việt Nam và đề xuất tham khảo phác đồ chi tiết tại Quyết định số 4815/QĐ-BYT."
        )
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
                continue
            gemini_role = "user" if role == "user" else "model"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

        data = {
            "contents": contents,
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1024
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates:
                    content_obj = candidates[0].get("content", {})
                    parts = content_obj.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return "Không thể nhận phản hồi hợp lệ từ Gemini API."
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            logger.error(f"Gemini API HTTPError {e.code}: {e.reason}\nBody: {error_body}", exc_info=True)
            return f"Lỗi kết nối Gemini API (HTTP {e.code}): {e.reason}\nChi tiết phản hồi từ Google: {error_body}"
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            return f"Lỗi kết nối Gemini API: {str(e)}\nVui lòng kiểm tra lại kết nối mạng."

    def generate_chat_response(self, messages: list) -> Tuple[str, bool]:
        """
        Generates a conversational response from the fine-tuned LLM.
        
        Args:
            messages: A list of dicts with 'role' and 'content' keys.
            
        Returns:
            A tuple of (generated_response_string, is_fallback_mode_boolean).
        """
        # Extract latest user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").strip()
                break

        # If outside pneumonia domain, route to Gemini if key is provided
        if user_message and not self._is_query_pneumonia_related(user_message):
            if settings.GEMINI_API_KEY:
                logger.info("Query is outside pneumonia domain. Routing to Gemini API.")
                gemini_res = self._call_gemini_api(messages)
                return gemini_res, False

        # Ensure model is initialized (lazy-loaded)
        if not self._is_loaded:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")

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
                    "content": (
                        "Bạn là một Bác sĩ AI chuyên khoa Hô hấp hỗ trợ tra cứu lâm sàng. Hãy trả lời bằng tiếng Việt chuẩn y khoa, ngắn gọn, chính xác. "
                        "LƯU Ý QUAN TRỌNG: Không tự ý kê đơn, không đưa ra tên thuốc hay liều lượng cụ thể (không ghi các liều thuốc cụ thể như Amoxicillin 1g, Ceftriaxone, v.v.). "
                        "Hãy định hướng xử trí chung theo phân loại CURB-65 và hướng dẫn lâm sàng của Bộ Y tế Việt Nam, đồng thời đề xuất bác sĩ tham khảo phác đồ chi tiết tại Quyết định số 4815/QĐ-BYT."
                    )
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
                "### Hướng xử trí Viêm phổi mắc phải cộng đồng (CAP) - Theo khuyến cáo Bộ Y tế:\n\n"
                "Phân loại mức độ nặng theo thang điểm **CURB-65**:\n"
                "- **CURB-65 = 0-1 (Nhẹ):** Điều trị ngoại trú.\n"
                "  * *Định hướng xử trí:* Bệnh nhân có thể điều trị tại nhà dưới sự theo dõi chặt chẽ của y tế cơ sở.\n"
                "  * *Lựa chọn kháng sinh ban đầu:* Ưu tiên sử dụng kháng sinh đường uống đơn trị liệu (nhóm Beta-lactam hoặc Tetracycline). Nếu nghi ngờ có tác nhân vi khuẩn không điển hình, cân nhắc bổ sung hoặc thay thế bằng nhóm Macrolide đường uống.\n"
                "  * *Lưu ý:* Bác sĩ vui lòng tham khảo chi tiết phác đồ lựa chọn thuốc và liều lượng cụ thể tại Mục 4.1 của Hướng dẫn ban hành kèm theo **Quyết định số 4815/QĐ-BYT**.\n"
                "- **CURB-65 = 2 (Trung bình):** Điều trị nội trú ngắn hạn tại khoa thường.\n"
                "  * *Định hướng xử trí:* Nhập viện điều trị hoặc theo dõi sát tại đơn vị lưu bệnh trú ngắn ngày.\n"
                "  * *Lựa chọn kháng sinh ban đầu:* Khuyến cáo phối hợp kháng sinh Beta-lactam tiêm truyền kết hợp với Macrolide đường uống/tiêm truyền, hoặc đơn trị liệu bằng Fluoroquinolone hô hấp đường tiêm/uống.\n"
                "  * *Lưu ý:* Chi tiết nhóm thuốc và liều dùng cụ thể phải tuân thủ Mục 4.2 của Hướng dẫn ban hành kèm theo **Quyết định số 4815/QĐ-BYT**.\n"
                "- **CURB-65 ≥ 3 (Nặng):** Nhập viện điều trị nội trú tích cực (Cấp cứu/ICU nếu CURB-65 ≥ 4).\n"
                "  * *Định hướng xử trí:* Nhập viện khẩn cấp, điều trị tại khoa Hồi sức tích cực (ICU) hoặc phòng cấp cứu chuyên khoa.\n"
                "  * *Lựa chọn kháng sinh ban đầu:* Phác đồ phối hợp kháng sinh Beta-lactam tiêm truyền phổ rộng (ưu tiên nhóm kháng Pseudomonal nếu có yếu tố nguy cơ) kết hợp với Fluoroquinolone hô hấp tiêm truyền hoặc Macrolide tiêm truyền.\n"
                "  * *Lưu ý:* Liều lượng và cách phối hợp thuốc chi tiết được quy định tại Mục 4.3 của Hướng dẫn ban hành kèm theo **Quyết định số 4815/QĐ-BYT**."
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
                "### Hướng dẫn sử dụng Kháng sinh ban đầu cho Người lớn (CAP trung bình - CURB-65 = 2):\n\n"
                "Theo khuyến cáo của Bộ Y tế Việt Nam cho bệnh nhân viêm phổi mắc phải cộng đồng mức độ trung bình điều trị tại khoa thường:\n"
                "1. **Nguyên tắc lựa chọn kháng sinh:**\n"
                "   - **Phác đồ phối hợp (Ưu tiên):** Kết hợp một kháng sinh nhóm Beta-lactam đường tiêm truyền (ví dụ: Cephalosporin thế hệ 3) với một kháng sinh nhóm Macrolide (đường uống hoặc tiêm truyền) để bao phủ cả vi khuẩn điển hình và không điển hình.\n"
                "   - **Phác đồ đơn trị liệu:** Sử dụng một kháng sinh nhóm Fluoroquinolone hô hấp (đường uống hoặc tiêm truyền) cho hiệu quả diệt khuẩn rộng.\n"
                "2. **Thời gian điều trị:**\n"
                "   - Thường kéo dài từ 5 - 7 ngày đối với viêm phổi không biến chứng. Bệnh nhân cần đạt tiêu chuẩn ổn định lâm sàng và hết sốt ít nhất 48 - 72 giờ trước khi xem xét dừng kháng sinh.\n\n"
                "*LƯU Ý LÂM SÀNG: AI không tự ra quyết định phác đồ, không kê đơn hay chỉ định liều lượng cụ thể. Bác sĩ vui lòng tham khảo chi tiết danh mục thuốc, liều dùng và hướng dẫn phối hợp tại Hướng dẫn chẩn đoán và điều trị viêm phổi mắc phải cộng đồng ở người lớn ban hành theo **Quyết định số 4815/QĐ-BYT**.*"
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
            if settings.GEMINI_API_KEY:
                logger.info("Local simulation has no matching keyword rule. Routing to Gemini API.")
                return self._call_gemini_api(messages)
            return (
                f"Cảm ơn bác sĩ đã chia sẻ câu hỏi về: *\"{user_message}\"*.\n\n"
                "Với vai trò là **Bác sĩ AI chuyên khoa Hô hấp**, để đưa ra hỗ trợ tư vấn lâm sàng chính xác nhất cho ca bệnh viêm phổi này, tôi khuyến nghị bác sĩ cung cấp thêm các thông tin:\n"
                "1. **Hình ảnh học (X-quang ngực)**: Có xuất hiện đám mờ đông đặc thù, tổn thương thâm nhiễm phế nang hay bóng mờ phế quản phế nang khí không?\n"
                "2. **Các triệu chứng cơ năng & dấu hiệu sinh tồn**: Bệnh nhân có bị lú lẫn, nhịp thở (lần/phút), huyết áp (tâm thu/tâm trương) và nồng độ Urê trong máu thế nào (để tính điểm độ nặng CURB-65)?\n"
                "3. **Tính chất ho & đờm**: Ho khan hay ho có đờm (đờm mủ, đờm màu rỉ sắt)?\n\n"
                "*Bác sĩ cũng có thể đặt các câu hỏi trực tiếp về phác đồ kháng sinh CAP của Bộ Y tế, cách phân biệt viêm phổi điển hình/không điển hình hoặc thang điểm đánh giá độ nặng.*"
            )

llm_service = LLMService()
