import os
import json
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Request
from starlette.concurrency import run_in_threadpool
from app.core.security import verify_api_key, limiter
from app.core.logging_config import logger, request_id_var
from app.utils.file_validator import validate_image_bytes

from app.ktp.v2.schemas_v2 import KTPOcrResponseV2, ConsensusResponseV2
from app.ktp.v3.service_v3 import process_ktp_image_v3, run_consensus_ocr_v3

router = APIRouter(prefix="/ktp/v3", tags=["ktp-v3-onnx"])

OCR_TIMEOUT_SECONDS = float(os.getenv("OCR_TIMEOUT_SECONDS", "8.0"))
UNREADABLE_IMAGE_DETAIL = (
    "Kualitas gambar KTP kurang jelas, buram, atau pencahayaan kurang optimal sehingga tidak dapat diproses. "
    "Silakan foto ulang KTP Anda dengan posisi tegak lurus dan pencahayaan yang cukup."
)


@router.post("/extract", response_model=KTPOcrResponseV2, dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def extract_ktp_v3(request: Request, file: UploadFile = File(...)):
    """
    Endpoint KTP OCR (Ekstraksi 15 Field Dukcapil).
    """
    req_id = request_id_var.get()
    logger.info(f"[OCR Extract] Request diterima - request_id={req_id} filename={file.filename}")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    validate_image_bytes(content)

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(process_ktp_image_v3, content),
            timeout=OCR_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"[OCR Extract Timeout] Ekstraksi melebihi batas waktu ({OCR_TIMEOUT_SECONDS}s) - request_id={req_id}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=UNREADABLE_IMAGE_DETAIL
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[OCR Extract Error] request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses OCR KTP: {str(e)}"
        )


@router.post("/validate", response_model=ConsensusResponseV2, dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def validate_ktp_v3(
    request: Request,
    file: UploadFile = File(...),
    mobile_data: str = Form(...),
):
    """
    Endpoint Consensus Validator (Membandingkan data Mobile dengan OCR).
    """
    req_id = request_id_var.get()
    logger.info(f"[OCR Validate] Request diterima - request_id={req_id} filename={file.filename}")

    try:
        if isinstance(mobile_data, str):
            mob_dict = json.loads(mobile_data)
        else:
            mob_dict = dict(mobile_data)
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

    validate_image_bytes(content)

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(run_consensus_ocr_v3, content, mob_dict),
            timeout=OCR_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"[OCR Validate Timeout] Validasi melebihi batas waktu ({OCR_TIMEOUT_SECONDS}s) - request_id={req_id}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=UNREADABLE_IMAGE_DETAIL
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[OCR Validate Error] request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memvalidasi KTP: {str(e)}"
        )
