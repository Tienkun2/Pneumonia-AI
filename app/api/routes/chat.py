from fastapi import APIRouter, Depends
from typing import Annotated
import logging
from app.schemas.response import ChatRequest, ChatResponse, ErrorResponse
from app.services.llm_service import LLMService, llm_service
from app.exceptions.custom_exceptions import PredictionException

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "",
    response_model=ChatResponse,
    summary="💬 Clinical AI Doctor Chatbot",
    description="Conversational endpoint powered by the fine-tuned LLM specialized in clinical respiratory advice.",
    responses={
        200: {"model": ChatResponse, "description": "Chat response generated successfully"},
        500: {"model": ErrorResponse, "description": "LLM Inference Error"}
    }
)
async def chat_with_doctor(
    request: ChatRequest,
    llm_serv: Annotated[LLMService, Depends(lambda: llm_service)]
):
    """
    Chat endpoint for interactive medical and clinical Q&A.
    """
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages.")
        
        # Convert Pydantic schemas to standard dictionaries for LLMService
        messages_list = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response_text, is_fallback = llm_serv.generate_chat_response(messages_list)
        
        return {
            "text": response_text,
            "fallback": is_fallback
        }
    except Exception as e:
        logger.error(f"Chat generation failed: {str(e)}", exc_info=True)
        raise PredictionException(f"Failed to generate chat response: {str(e)}")
