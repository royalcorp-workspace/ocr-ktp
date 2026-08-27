import os
import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

load_dotenv()

from app.core.logging_config import logger, request_id_var
from app.core.security import limiter
from app.api import ktp_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8011")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.info(f"Menginisialisasi microservice ocr-ktp - host={host} port={port} log_level={log_level}")
    
    # Eager Model Warmup: load PaddleOCR models into memory during container startup
    try:
        from app.ktp.v2.paddle_engine import PaddleEngineV2
        engine = PaddleEngineV2()
        engine.warmup()
        logger.info("PaddleOCR V2 Engine berhasil di-warmup pada startup.")
    except Exception as e:
        logger.warning(f"Gagal melakukan warmup PaddleOCR V2 Engine: {e}")

    yield
    logger.info("Menghentikan microservice ocr-ktp")


app = FastAPI(
    title="OCR KTP Microservice",
    description="Standalone Microservice FastAPI untuk ekstraksi teks & structured data KTP-el Indonesia (OCR).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(req_id)
    try:
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        # HTTP Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    finally:
        request_id_var.reset(token)


from app.api import v1_routes, v2_routes
app.include_router(ktp_routes.router)
app.include_router(v1_routes.router)
app.include_router(v2_routes.router)


@app.get("/health", tags=["health"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "service": "ocr-ktp",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    host_val = os.getenv("HOST", "0.0.0.0")
    port_val = int(os.getenv("PORT", "8011"))
    uvicorn.run("main:app", host=host_val, port=port_val, reload=True)