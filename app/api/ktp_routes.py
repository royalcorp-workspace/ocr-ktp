import re
import time
import json
from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Request
from app.core.security import verify_api_key, limiter
from app.core.logging_config import logger, request_id_var
from app.ktp.service import process_ktp_image
from app.ktp.consensus import run_consensus_ocr
from app.ktp.schemas import (
    KTPOcrResponse,
    FieldWithConfidence,
    MobileOCRInput,
    ConsensusResponse,
    ValidatedField,
    QualityMetrics,
)

from app.utils.file_validator import validate_image_bytes

router = APIRouter(prefix="/ktp", tags=["ktp"])


from app.ktp.confidence import calculate_field_confidence


def _make_field_response(
    field_name: str,
    val: str | None,
    best_score: int,
    best_word_conf_map: dict | None,
    raw_text: str | None,
    all_fields: dict | None = None,
    min_confidence_threshold: float = 30.0,
) -> FieldWithConfidence:
    if not val:
        return FieldWithConfidence(value=None, confidence=0.0)
    conf = calculate_field_confidence(
        field_name, val, best_score, best_word_conf_map, raw_text=raw_text, all_fields=all_fields
    )
    if conf < min_confidence_threshold:
        return FieldWithConfidence(value=None, confidence=0.0)
    return FieldWithConfidence(value=val, confidence=conf)


@router.post("/extract", response_model=KTPOcrResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def extract_ktp(request: Request, file: UploadFile = File(...)):
    """
    Endpoint untuk mengekstrak informasi terstruktur dari Kartu Tanda Penduduk (KTP-el) Indonesia (OCR).
    Menerima file gambar (JPEG, PNG, WebP, BMP) dan mengembalikan 4 field utama beserta confidence score.
    """
    req_id = request_id_var.get()
    start_time = time.perf_counter()
    logger.info(f"KTP OCR request diterima - request_id={req_id} filename={file.filename} content_type={file.content_type}")

    try:
        content = await file.read()
    except Exception as e:
        logger.warning(f"KTP OCR gagal validasi - request_id={req_id} filename={file.filename} reason=read_bytes_failed error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    # Validasi binary image (Magic Bytes & Decoding)
    validate_image_bytes(content)

    logger.info(f"KTP OCR validation berhasil - request_id={req_id} filename={file.filename}")
    logger.info(f"KTP OCR processing dimulai - request_id={req_id}")

    try:
        raw_text, full_result, best_score, best_word_conf_map = await run_in_threadpool(process_ktp_image, content)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"KTP OCR berhasil - request_id={req_id} filename={file.filename} duration_ms={duration_ms}")

        all_fields = {
            "nik": full_result.nik,
            "nama": full_result.nama,
            "tempat_lahir": full_result.tempat_lahir,
            "tanggal_lahir": full_result.tanggal_lahir,
        }

        return KTPOcrResponse(
            nik=_make_field_response(
                "nik", full_result.nik, best_score, best_word_conf_map, raw_text=raw_text, all_fields=all_fields
            ),
            nama=_make_field_response(
                "nama", full_result.nama, best_score, best_word_conf_map, raw_text=raw_text, all_fields=all_fields
            ),
            tempat_lahir=_make_field_response(
                "tempat_lahir", full_result.tempat_lahir, best_score, best_word_conf_map, raw_text=raw_text, all_fields=all_fields
            ),
            tanggal_lahir=_make_field_response(
                "tanggal_lahir", full_result.tanggal_lahir, best_score, best_word_conf_map, raw_text=raw_text, all_fields=all_fields
            ),
        )
    except ValueError as e:

        logger.warning(f"KTP OCR gagal validasi - request_id={req_id} reason=invalid_image error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"KTP OCR engine error - request_id={req_id} error_type={e.__class__.__name__} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal memproses gambar KTP melalui layanan OCR."
        )


@router.post("/validate", response_model=ConsensusResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def validate_ktp(
    request: Request,
    file: UploadFile = File(...),
    mobile_data: str = Form(...),
):
    """
    Endpoint Weighted Consensus OCR (Validator & Auto-Corrector KTP).

    Menerima gambar KTP beserta data OCR dari Mobile (Sistem A) yang dilengkapi confidence score.
    Menjalankan multi-kandidat Tesseract OCR secara paralel, kemudian melakukan Weighted Voting
    per-field antara data Mobile dan semua hasil Tesseract untuk menentukan nilai final.

    - **file**: Gambar KTP (JPEG/PNG/WebP), diproses di RAM.
    - **mobile_data**: JSON string dengan format per-field:
      ```json
      {
        "nik": { "value": "3217061202990002", "confidence": 91.2 },
        "nama": { "value": "Alghany Kennedy Adam", "confidence": 84.5 },
        "tempat_lahir": { "value": "Bandung", "confidence": 78.0 },
        "tanggal_lahir": { "value": "12-02-1999", "confidence": 80.0 }
      }
      ```
    """
    req_id = request_id_var.get()
    start_time = time.perf_counter()
    logger.info(f"KTP VALIDATE (Consensus) request diterima - request_id={req_id} filename={file.filename}")

    # --- 1. Parse & Normalize mobile_data JSON ---
    try:
        from app.ktp.normalizer import GlobalPayloadNormalizer
        norm_flat = GlobalPayloadNormalizer.normalize(mobile_data)
        
        raw_dict = {}
        if isinstance(mobile_data, str):
            try:
                raw_dict = json.loads(mobile_data)
            except Exception:
                pass
        elif isinstance(mobile_data, dict):
            raw_dict = mobile_data

        mobile_input_dict = {}
        for f in ["nik", "nama", "tempat_lahir", "tanggal_lahir"]:
            val = norm_flat.get(f)
            conf = 0.0
            if isinstance(raw_dict, dict):
                field_raw = raw_dict.get(f)
                if isinstance(field_raw, dict) and "confidence" in field_raw:
                    try:
                        conf = float(field_raw["confidence"])
                    except (ValueError, TypeError):
                        pass
            if val is not None:
                mobile_input_dict[f] = {"value": val, "confidence": conf}
            else:
                mobile_input_dict[f] = None
                
        mobile_input = MobileOCRInput(**mobile_input_dict)
    except Exception as e:
        logger.warning(f"KTP VALIDATE gagal parse mobile_data - request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Format mobile_data tidak valid. Kirim JSON string yang benar. Error: {str(e)}"
        )

    # --- 2. Validasi file gambar ---
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    # Validasi binary image (Magic Bytes & Decoding)
    validate_image_bytes(content)

    # --- 3. Jalankan Consensus OCR Pipeline ---
    logger.info(f"KTP VALIDATE (Consensus) processing dimulai - request_id={req_id}")
    try:
        result = await run_in_threadpool(
            run_consensus_ocr,
            content,
            mobile_input.model_dump(),
        )
    except Exception as e:
        logger.error(f"KTP VALIDATE engine error - request_id={req_id} error_type={e.__class__.__name__} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal memproses gambar KTP melalui layanan Consensus OCR."
        )

    # --- 4. Build response ---
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        f"KTP VALIDATE (Consensus) selesai - request_id={req_id} filename={file.filename} "
        f"duration_ms={duration_ms} success={result.get('success')}"
    )

    validated_data = {}
    for field_name, field_val in result.get("data", {}).items():
        validated_data[field_name] = ValidatedField(**field_val)

    quality_raw = result.get("quality", {})

    return ConsensusResponse(
        success=result.get("success", False),
        data=validated_data,
        quality=QualityMetrics(**quality_raw),
        warnings=result.get("warnings", []),
    )
