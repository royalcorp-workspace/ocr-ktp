import time
import json
from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Request
from app.core.security import verify_api_key, limiter
from app.core.logging_config import logger, request_id_var

from app.ktp.v1.schemas_v1 import KTPOcrResponseV1, ConsensusResponseV1, MobileDataInputV1, QualityMetricsV1
from app.ktp.v1.service_v1 import process_ktp_image_v1
from app.ktp.v1.consensus_v1 import run_consensus_ocr_v1

from app.utils.file_validator import validate_image_bytes

router = APIRouter(prefix="/ktp/v1", tags=["ktp-v1"])

@router.post("/extract", response_model=KTPOcrResponseV1, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def extract_ktp_v1(request: Request, file: UploadFile = File(...)):
    """
    Endpoint KTP OCR V1 (14 Field).
    Menggunakan Targeted OCR (ROI) yang dipadukan dengan General OCR Fallback.
    """
    req_id = request_id_var.get()
    logger.info(f"[V1] KTP Extract request diterima - request_id={req_id}")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    # Validasi binary image (Magic Bytes & Decoding)
    validate_image_bytes(content)

    try:
        response_data = await run_in_threadpool(process_ktp_image_v1, content)
        return response_data
    except Exception as e:
        logger.error(f"[V1] Extract Error - request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses gambar KTP: {str(e)}"
        )

@router.post("/validate", response_model=ConsensusResponseV1, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def validate_ktp_v1(
    request: Request,
    file: UploadFile = File(...),
    mobile_data: str = Form(...),
):
    """
    Endpoint Consensus Validator V1 (14 Field).
    Membandingkan data Mobile dengan General OCR, lalu fallback dengan Targeted ROI OCR.
    """
    req_id = request_id_var.get()
    logger.info(f"[V1] KTP Validate request diterima - request_id={req_id}")

    try:
        from app.ktp.normalizer import GlobalPayloadNormalizer
        mobile_input = GlobalPayloadNormalizer.to_mobile_input_v1(mobile_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Format mobile_data tidak valid. Error: {str(e)}"
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    # Validasi binary image (Magic Bytes & Decoding)
    validate_image_bytes(content)

    try:
        result = await run_in_threadpool(run_consensus_ocr_v1, content, mobile_input.model_dump())
        return ConsensusResponseV1(
            success=result.get("success", False),
            data=result.get("data", {}),
            quality=QualityMetricsV1(**result.get("quality", {})),
            warnings=result.get("warnings", []),
        )
    except Exception as e:
        logger.error(f"[V1] Validate Error - request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memvalidasi KTP: {str(e)}"
        )
