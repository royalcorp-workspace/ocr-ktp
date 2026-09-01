import json
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Request
from starlette.concurrency import run_in_threadpool
from app.core.security import verify_api_key, limiter
from app.core.logging_config import logger, request_id_var
from app.utils.file_validator import validate_image_bytes

from app.ktp.v2.schemas_v2 import KTPOcrResponseV2, ConsensusResponseV2
from app.ktp.v3.service_v3 import process_ktp_image_v3, run_consensus_ocr_v3

router = APIRouter(prefix="/ktp/v3", tags=["ktp-v3-onnx"])

@router.post("/extract", response_model=KTPOcrResponseV2, dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def extract_ktp_v3(request: Request, file: UploadFile = File(...)):
    """
    Endpoint KTP OCR V3 (ONNX Runtime Standalone).
    Ekstraksi 15 Field Dukcapil menggunakan PP-OCRv4 ONNX Runtime berkecepatan tinggi.
    """
    req_id = request_id_var.get()
    logger.info(f"[V3 ONNX] KTP Extract request diterima - request_id={req_id} filename={file.filename}")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    validate_image_bytes(content)

    try:
        result = await run_in_threadpool(process_ktp_image_v3, content)
        return result
    except Exception as e:
        logger.error(f"[V3 ONNX] Extract Error - request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses OCR KTP V3 ONNX: {str(e)}"
        )

@router.post("/validate", response_model=ConsensusResponseV2, dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def validate_ktp_v3(
    request: Request,
    file: UploadFile = File(...),
    mobile_data: str = Form(...),
):
    """
    Endpoint Consensus Validator V3 (ONNX Runtime Standalone).
    Membandingkan data Mobile dengan ONNX OCR V3 (source: 'OCR' atau 'MOBILE').
    """
    req_id = request_id_var.get()
    logger.info(f"[V3 ONNX] KTP Validate request diterima - request_id={req_id} filename={file.filename}")

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
        result = await run_in_threadpool(run_consensus_ocr_v3, content, mob_dict)
        return result
    except Exception as e:
        logger.error(f"[V3 ONNX] Validate Error - request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memvalidasi KTP V3 ONNX: {str(e)}"
        )
