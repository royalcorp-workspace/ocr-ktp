import os
import json
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from starlette.concurrency import run_in_threadpool

from app.core.security import verify_api_key, limiter
from app.core.logging_config import logger, request_id_var
from app.utils.file_validator import validate_image_bytes
from app.ktp.v4.schemas_v4 import KTPValidationResponseV4
from app.ktp.v4.service_v4 import run_ktp_validation_v4

router = APIRouter(prefix="/ktp/v4", tags=["ktp-v4-validation"])

OCR_TIMEOUT_SECONDS = float(os.getenv("OCR_TIMEOUT_SECONDS", "8.0"))
UNREADABLE_IMAGE_DETAIL = (
    "Kualitas gambar KTP kurang jelas, buram, atau pencahayaan kurang optimal sehingga tidak dapat diproses. "
    "Silakan foto ulang KTP Anda dengan posisi tegak lurus dan pencahayaan yang cukup."
)


@router.post("/validate", response_model=KTPValidationResponseV4, dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def validate_ktp_v4(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Endpoint KTP OCR V4 (Pop-up Validation & Multi-Field Comparison):
    - Menerima gambar KTP (file) dan data pembanding (db_data).
    - Mendukung format db_data fleksibel: JSON string dalam form-field ('db_data', 'form_data', 'data')
      maupun form-fields individual (misal: 'nik', 'nama', 'agama', dsb).
    - Menghasilkan response kanonikal perbandingan untuk modal/pop-up validasi di frontend.
    """
    req_id = request_id_var.get()
    logger.info(f"[OCR V4 Validate] Request diterima - request_id={req_id} filename={file.filename}")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    validate_image_bytes(content)

    # Parsing fleksibel untuk input data pembanding (DB / Form)
    form_data = await request.form()
    db_dict: Dict[str, Any] = {}

    json_field_keys = ["db_data", "data", "form_data", "mobile_data", "input_data"]
    for jk in json_field_keys:
        if jk in form_data:
            field_val = form_data[jk]
            if isinstance(field_val, str) and field_val.strip():
                try:
                    parsed = json.loads(field_val)
                    if isinstance(parsed, dict):
                        db_dict.update(parsed)
                        break
                except Exception as e:
                    logger.warning(f"[OCR V4 Validate] Gagal parse JSON string dari field '{jk}': {e}")

    # Tambahkan atau lengkapi dengan field individual non-file
    for k, v in form_data.items():
        if k not in json_field_keys and not isinstance(v, UploadFile):
            if k not in db_dict:
                db_dict[k] = v

    if not db_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Data pembanding (db_data atau field form individual) tidak ditemukan atau tidak valid."
        )

    timeout_sec = float(os.getenv("OCR_TIMEOUT_SECONDS", "15.0"))
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(run_ktp_validation_v4, content, db_dict),
            timeout=timeout_sec
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"[OCR V4 Timeout] Validasi melebihi batas waktu ({timeout_sec}s) - request_id={req_id}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=UNREADABLE_IMAGE_DETAIL
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[OCR V4 Error] request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses validasi OCR KTP V4: {str(e)}"
        )
