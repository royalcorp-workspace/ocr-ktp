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

router = APIRouter(prefix="/ktp", tags=["ktp"])


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Menghitung Levenshtein edit distance antara dua string."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    dp = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        new_dp = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            new_dp[j + 1] = min(
                dp[j + 1] + 1,      # deletion
                new_dp[j] + 1,      # insertion
                dp[j] + cost        # substitution
            )
        dp = new_dp
    return dp[-1]


def _find_raw_candidate(
    field_name: str,
    val_str: str,
    raw_text: str | None,
    word_conf_map: dict | None
) -> str | None:
    """
    Mencari baris/token mentah OCR (sebelum dikoreksi) dari raw_text atau word_conf_map
    yang paling relevan dengan val_str untuk evaluasi perbandingan.
    """
    if not raw_text and not word_conf_map:
        return None

    raw_text_clean = raw_text or ""
    lines = [line.strip() for line in raw_text_clean.splitlines() if line.strip()]

    if field_name == "nik":
        clean_val = re.sub(r'[^0-9]', '', val_str)
        cands = []
        for line in lines:
            line_clean = re.sub(r'[^A-Za-z0-9]', '', line).upper()
            for token in line.split():
                t_clean = re.sub(r'[^A-Za-z0-9]', '', token).upper()
                if len(t_clean) >= 14:
                    cands.append(t_clean)
            if len(line_clean) >= 14 and sum(1 for c in line_clean if c.isdigit() or c in 'OOLISZBQ') >= 10:
                cands.append(line_clean)

        if word_conf_map:
            for k in word_conf_map.keys():
                k_clean = re.sub(r'[^A-Za-z0-9]', '', k).strip().upper()
                if len(k_clean) >= 14:
                    cands.append(k_clean)

        if cands:
            return min(cands, key=lambda c: _levenshtein_distance(c, clean_val))
        return None

    elif field_name == "nama":
        val_clean = re.sub(r'[^A-Z\s]', '', val_str.upper()).strip()
        best_line = None
        min_dist = float('inf')

        for line in lines:
            line_up = line.upper()
            if any(lbl in line_up for lbl in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "GOL. DARAH", "AGAMA"]):
                continue
            line_proc = re.sub(r'^\s*NAMA\s*[:\.]?\s*', '', line_up)
            line_clean = re.sub(r'[^A-Z\s]', '', line_proc).strip()
            if not line_clean or len(line_clean) < 2:
                continue

            # Jika nama ada di dalam baris (misal OCR menggabungkan baris NAMA dan Tempat Lahir)
            if val_clean in line_clean:
                best_line = val_clean
                min_dist = 0
                break

            dist = _levenshtein_distance(line_clean, val_clean)
            if dist < min_dist:
                min_dist = dist
                best_line = line_clean

        return best_line

    elif field_name == "tanggal_lahir":
        clean_val = re.sub(r'[^0-9]', '', val_str)
        cands = []
        for line in lines:
            for t in line.split():
                t_clean = re.sub(r'[^0-9]', '', t)
                if len(t_clean) == 8:
                    cands.append(t_clean)
        if cands:
            return min(cands, key=lambda c: _levenshtein_distance(c, clean_val))
        return None

    elif field_name == "tempat_lahir":
        val_clean = re.sub(r'[^A-Z]', '', val_str.upper()).strip()
        cands = []
        for line in lines:
            for t in line.split():
                t_clean = re.sub(r'[^A-Z]', '', t).upper()
                if abs(len(t_clean) - len(val_clean)) <= 3 and len(t_clean) >= 3:
                    cands.append(t_clean)
        if cands:
            return min(cands, key=lambda c: _levenshtein_distance(c, val_clean))
        return None

    return None


def _calculate_field_confidence(
    field_name: str,
    value: str | None,
    base_score: int,
    word_conf_map: dict | None = None,
    raw_text: str | None = None,
    all_fields: dict | None = None,
) -> float:
    """
    Kalkulasi confidence score per-field secara TERKALIBRASI (Layered Gated Confidence System).
    """
    if not value or not str(value).strip():
        return 0.0

    val_str = str(value).strip().upper()
    tokens = [re.sub(r'[^A-Z0-9]', '', t.strip()) for t in val_str.split() if t.strip()]
    tokens = [t for t in tokens if t]

    # 1. Base Score calculation
    tess_confs = []
    if word_conf_map:
        for t in tokens:
            if t in word_conf_map:
                tess_confs.append(word_conf_map[t])

    if tess_confs:
        avg_tess_conf = sum(tess_confs) / len(tess_confs)
    else:
        avg_tess_conf = float(min(85.0, max(50.0, base_score)))

    raw_base = (0.70 * avg_tess_conf) + (0.30 * float(min(95.0, base_score)))

    bonus = 0.0
    penalty = 0.0
    gated_cap = 99.0

    raw_cand = _find_raw_candidate(field_name, val_str, raw_text, word_conf_map)

    # 2. Field-Specific Rules, Penalties & Caps
    if field_name == "nik":
        clean_nik = re.sub(r'[^0-9]', '', val_str)
        is_valid_len = (len(clean_nik) == 16)
        is_valid_struct = False

        if is_valid_len:
            from app.ktp.extractor.validators import validate_nik_structure
            is_valid_struct = validate_nik_structure(clean_nik, raw_text or "")

        dob_matched = False
        dob_mismatched = False
        if all_fields and is_valid_len:
            dob_val = all_fields.get("tanggal_lahir")
            if dob_val:
                dob_clean = str(dob_val).strip()
                dob_match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', dob_clean)
                if dob_match:
                    f_day = int(dob_match.group(1))
                    f_month = int(dob_match.group(2))
                    f_year_two = dob_match.group(3)[2:]

                    nik_day = int(clean_nik[6:8])
                    if nik_day > 40:
                        nik_day -= 40
                    nik_month = int(clean_nik[8:10])
                    nik_year_two = clean_nik[10:12]

                    if nik_day == f_day and nik_month == f_month and nik_year_two == f_year_two:
                        dob_matched = True
                    else:
                        dob_mismatched = True

        if is_valid_struct:
            bonus += 12.0
            if dob_matched:
                bonus += 8.0
        elif is_valid_len:
            bonus += 2.0
        else:
            gated_cap = min(gated_cap, 45.0)

        if dob_mismatched:
            penalty += 25.0
            gated_cap = min(gated_cap, 55.0)

        from app.ktp.extractor.validators import PROVINCE_CODES
        if is_valid_len and clean_nik[:2] not in PROVINCE_CODES.values():
            penalty += 20.0
            gated_cap = min(gated_cap, 50.0)

        if raw_cand:
            clean_cand = re.sub(r'[^0-9]', '', raw_cand)
            non_digit_count = sum(1 for c in raw_cand if not c.isdigit())
            edit_dist = _levenshtein_distance(clean_cand, clean_nik)

            if not is_valid_struct:
                if non_digit_count > 0:
                    penalty += (5.0 * non_digit_count)
                if edit_dist >= 3:
                    penalty += 30.0
                    gated_cap = min(gated_cap, 50.0)

    elif field_name == "nama":
        if len(tokens) >= 2:
            bonus += 5.0

        digit_in_val = sum(1 for c in val_str if c.isdigit())
        if digit_in_val > 0:
            penalty += (15.0 * digit_in_val)
            gated_cap = min(gated_cap, 50.0)

        if raw_cand:
            raw_cand_clean = re.sub(r'[^A-Z\s]', '', raw_cand).strip()
            val_clean = re.sub(r'[^A-Z\s]', '', val_str).strip()
            edit_dist = _levenshtein_distance(raw_cand_clean, val_clean)
            max_len = max(len(val_clean), 1)
            rel_dist = edit_dist / max_len

            if edit_dist == 0 or val_clean in raw_cand_clean:
                bonus += 5.0
            elif rel_dist <= 0.20:
                penalty += (3.0 * edit_dist)
            elif rel_dist <= 0.40:
                penalty += 10.0
                gated_cap = min(gated_cap, 80.0)
            else:
                penalty += 20.0
                gated_cap = min(gated_cap, 65.0)

    elif field_name == "tanggal_lahir":
        if re.match(r'^\d{2}-\d{2}-\d{4}$', val_str):
            bonus += 10.0
            try:
                d, m, y = map(int, val_str.split('-'))
                if not (1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2026):
                    gated_cap = min(gated_cap, 45.0)
            except ValueError:
                gated_cap = min(gated_cap, 45.0)
        else:
            gated_cap = min(gated_cap, 50.0)

        if all_fields and all_fields.get("nik"):
            nik_clean = re.sub(r'[^0-9]', '', str(all_fields.get("nik")))
            if len(nik_clean) == 16:
                dob_match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', val_str)
                if dob_match:
                    f_day = int(dob_match.group(1))
                    f_month = int(dob_match.group(2))
                    f_year_two = dob_match.group(3)[2:]

                    nik_day = int(nik_clean[6:8])
                    if nik_day > 40:
                        nik_day -= 40
                    nik_month = int(nik_clean[8:10])
                    nik_year_two = nik_clean[10:12]

                    if not (nik_day == f_day and nik_month == f_month and nik_year_two == f_year_two):
                        penalty += 25.0
                        gated_cap = min(gated_cap, 55.0)

    elif field_name == "tempat_lahir":
        from app.ktp.extractor.common import INDONESIAN_CITIES
        if val_str in INDONESIAN_CITIES:
            bonus += 10.0
        else:
            gated_cap = min(gated_cap, 75.0)

        if raw_cand:
            raw_cand_clean = re.sub(r'[^A-Z]', '', raw_cand).strip()
            val_clean = re.sub(r'[^A-Z]', '', val_str).strip()
            if raw_cand_clean != val_clean:
                edit_dist = _levenshtein_distance(raw_cand_clean, val_clean)
                if edit_dist >= 1:
                    penalty += (4.0 * edit_dist)

    # 3. Final score calculation with penalties & gated caps
    calculated_score = raw_base + bonus - penalty
    final_conf = min(gated_cap, max(10.0, calculated_score))

    return round(final_conf, 1)


def _make_field_response(
    field_name: str,
    val: str | None,
    best_score: int,
    best_word_conf_map: dict | None,
    raw_text: str | None,
    all_fields: dict | None,
    min_confidence_threshold: float = 35.0
) -> FieldWithConfidence:
    if not val or not str(val).strip():
        return FieldWithConfidence(value=None, confidence=0.0)
    conf = _calculate_field_confidence(
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
    Menerima file gambar (JPEG, PNG, WebP) dan mengembalikan 4 field utama beserta confidence score.
    """
    req_id = request_id_var.get()
    start_time = time.perf_counter()
    logger.info(f"KTP OCR request diterima - request_id={req_id} filename={file.filename} content_type={file.content_type}")

    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else ""
    if file.content_type not in allowed_types and ext not in ["jpg", "jpeg", "png", "webp"]:
        logger.warning(f"KTP OCR gagal validasi - request_id={req_id} filename={file.filename} reason=unsupported_media_type content_type={file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Tipe file tidak didukung. Harap unggah gambar JPEG, PNG, atau WebP."
        )

    try:
        content = await file.read()
    except Exception as e:
        logger.warning(f"KTP OCR gagal validasi - request_id={req_id} filename={file.filename} reason=read_bytes_failed error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    max_file_size = 10 * 1024 * 1024  # 10 MB
    if len(content) > max_file_size:
        logger.warning(f"KTP OCR gagal validasi - request_id={req_id} filename={file.filename} reason=file_too_large size_bytes={len(content)}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Payload Too Large. Ukuran file terlalu besar, maksimal 10 MB."
        )

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

    # --- 1. Parse mobile_data JSON ---
    try:
        mobile_dict = json.loads(mobile_data)
        mobile_input = MobileOCRInput(**mobile_dict)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"KTP VALIDATE gagal parse mobile_data - request_id={req_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Format mobile_data tidak valid. Kirim JSON string yang benar. Error: {str(e)}"
        )

    # --- 2. Validasi file gambar ---
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else ""
    if file.content_type not in allowed_types and ext not in ["jpg", "jpeg", "png", "webp"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Tipe file tidak didukung. Harap unggah gambar JPEG, PNG, atau WebP."
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gagal membaca bytes dari file yang diunggah."
        )

    max_file_size = 10 * 1024 * 1024  # 10 MB
    if len(content) > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Payload Too Large. Ukuran file terlalu besar, maksimal 10 MB."
        )

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
