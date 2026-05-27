from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import uuid
from app.api.routes import predict, health, symptoms, report, chat
from app.core.config import settings
from app.dependencies.model_loader import model_loader

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s",
)

# Custom log filter for request_id
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'SYSTEM')
        return True

logging.getLogger().addFilter(RequestIdFilter())
logger = logging.getLogger(__name__)

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Pneumonia AI Diagnosis API",
    description=(
        "## 🏥 Welcome to the Pneumonia AI Diagnosis Portal\n\n"
        "This tool helps healthcare professionals identify symptoms and analyze Chest X-ray images "
        "with the power of Artificial Intelligence. It is built to be combined with medical expertise "
        "to deliver faster and more accurate assessments.\n\n"
        "### 📋 How to use this Diagnostic Portal:\n"
        "1. **📸 Upload X-ray**: Submit a clean Chest X-ray image (supported: JPG, PNG).\n"
        "2. **📝 Describe Symptoms**: List what the patient is feeling (from the recognized list of 10).\n"
        "3. **✅ Receive Assessment**: Get a detailed report on risk levels (LOW, MEDIUM, or HIGH).\n\n"
        "### 🚦 Available Endpoints:\n"
        "- `POST /api/v1/predict`: Main diagnosis endpoint (Vision + Clinical).\n"
        "- `GET /api/v1/symptoms`: Full list of symptoms recognized by our Clinical AI.\n"
        "- `GET /api/v1/health`: System and hardware status.\n\n"
        "--- \n"
        "**Note**: This is an AI-assisted diagnostic tool. Results must be reviewed by a certified medical professional."
    ),
    version="1.0.0",
    contact={
        "name": "Pneumonia AI Support Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS (Essential for Frontend Integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all. In production, specify origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check logs."},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP Error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Middleware: Request ID & Timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    # Inject request_id into logs for this request context
    # This is a simple approach; for full context propagation, use ContextVars
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include Routes with Versioning
app.include_router(health.router, prefix=f"{settings.API_V1_STR}", tags=["Health"])
app.include_router(symptoms.router, prefix=f"{settings.API_V1_STR}", tags=["Diagnosis"])
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}", tags=["Diagnosis"])
app.include_router(report.router, prefix=f"{settings.API_V1_STR}/report", tags=["Diagnosis Report"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["Chatbot"])

@app.on_event("startup")
async def startup_event():
    logger.info("Service is starting up...")
    if not model_loader.is_ready:
        logger.warning("Models are not fully initialized yet.")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=2)

