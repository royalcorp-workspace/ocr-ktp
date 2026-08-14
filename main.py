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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(req_id)
    try:
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        request_id_var.reset(token)


app.include_router(ktp_routes.router)


@app.get("/health", tags=["health"])
async def health_check():
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