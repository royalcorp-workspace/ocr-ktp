import time
import cv2
cv2.setNumThreads(1)
import numpy as np
from app.ktp.schemas import KTPData
from app.ktp.extractor import KTPExtractor
from app.ktp.preprocessing import normalize_image_size, auto_orient_image
from app.ktp.engine import run_candidates_tiered
from app.core.logging_config import ktp_logger as logger


def process_ktp_image(image_bytes: bytes) -> tuple:
    """
    Memproses byte gambar KTP-el Indonesia:
    normalize size -> auto-orient -> tiered parallel OCR execution -> pilih terbaik -> ekstrak field.
    """
    if not image_bytes:
        raise ValueError("Image bytes must not be empty.")

    t_start = time.perf_counter()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image. Invalid or corrupted image format.")

    image = normalize_image_size(image)
    image = auto_orient_image(image)

    t_ocr_start = time.perf_counter()
    res_tuple = run_candidates_tiered(image)
    if len(res_tuple) == 6:
        best_raw_text, best_candidate_name, best_score, candidates_executed, best_word_conf_map, all_tier_results = res_tuple
    else:
        best_raw_text, best_candidate_name, best_score, candidates_executed, best_word_conf_map = res_tuple
        all_tier_results = []
    t_ocr_end = time.perf_counter()

    ocr_duration_ms = round((t_ocr_end - t_ocr_start) * 1000, 2)
    total_pipeline_ms = round((t_ocr_end - t_start) * 1000, 2)

    logger.info(
        f"TIMING BREAKDOWN - run_candidates_tiered={ocr_duration_ms}ms, "
        f"total_pipeline={total_pipeline_ms}ms, "
        f"candidates_executed={candidates_executed}"
    )

    logger.info(f"KTP OCR candidate terpilih - name={best_candidate_name} score={best_score}")

    extractor = KTPExtractor()
    parsed_data = extractor.extract(best_raw_text)

    # 1. NIK-DOB Read-Only Consistency Check
    if parsed_data.nik and parsed_data.tanggal_lahir:
        from app.ktp.extractor.validators import is_nik_consistent_with_birthdate
        is_consistent = is_nik_consistent_with_birthdate(parsed_data.nik, parsed_data.tanggal_lahir, parsed_data.jenis_kelamin)
        logger.info(f"[SERVICE] NIK-DOB Consistency Check: {is_consistent} (NIK: {parsed_data.nik})")

    # 2. Multi-Candidate NIK Voting (Character-Level Consensus)
    if all_tier_results and parsed_data.nik and len(parsed_data.nik) == 16:
        from app.ktp.extractor.validators import vote_nik_character_level
        raw_texts = [res.get("raw_text", "") for res in all_tier_results]
        voted_nik = vote_nik_character_level(
            parsed_data.nik, 
            raw_texts, 
            tanggal_lahir=parsed_data.tanggal_lahir, 
            jenis_kelamin=parsed_data.jenis_kelamin
        )
        if voted_nik != parsed_data.nik:
            logger.info(f"NIK Multi-Candidate Voting consensus applied: '{parsed_data.nik}' -> '{voted_nik}'")
            parsed_data.nik = voted_nik

    return best_raw_text, parsed_data, best_score, best_word_conf_map



# Field-field yang akan direkonsiliasi. Tambahkan entry baru di sini
# untuk memperluas cakupan validasi (misal: "alamat", "tempat_lahir").
RECONCILE_FIELDS = ["nik", "nama"]


def reconcile_ktp_data(mobile_data: dict, engine_data: dict) -> tuple[dict, bool, list]:
    """
    Membandingkan data OCR dari Mobile (Sistem A) dengan hasil Engine OCR internal (Sistem C).

    Logika rekonsiliasi per-field:
    - Jika engine memiliki hasil (non-None/non-empty) DAN hasilnya BERBEDA
      dengan mobile_data, maka timpa dengan data engine (override).
    - Jika engine tidak memiliki hasil (None/empty), pertahankan data mobile.

    Args:
        mobile_data: Dict dari hasil OCR Mobile (misal: {"nik": "3204...", "nama": "DEDEM"}).
        engine_data: Dict dari hasil Engine OCR internal (misal: {"nik": "3204...", "nama": "DEDEN"}).

    Returns:
        Tuple berisi:
        - reconciled: Dict data final yang sudah direkonsiliasi.
        - is_corrected: True jika ada minimal 1 field yang dikoreksi.
        - corrections: List of dict detail koreksi per-field yang berubah.
    """
    reconciled = {}
    corrections = []

    for field in RECONCILE_FIELDS:
        mobile_val = mobile_data.get(field)
        engine_val = engine_data.get(field)

        # Normalisasi: anggap empty string sama dengan None
        mobile_val_norm = (mobile_val or "").strip() or None
        engine_val_norm = (engine_val or "").strip() or None

        if engine_val_norm and engine_val_norm != mobile_val_norm:
            # Engine punya hasil dan BERBEDA -> override dengan data engine
            reconciled[field] = engine_val_norm
            corrections.append({
                "field": field,
                "mobile_value": mobile_val_norm,
                "corrected_value": engine_val_norm,
            })
            logger.info(
                f"RECONCILE OVERRIDE: field='{field}' "
                f"mobile='{mobile_val_norm}' -> engine='{engine_val_norm}'"
            )
        else:
            # Engine tidak punya hasil atau sama -> pertahankan data mobile
            reconciled[field] = mobile_val_norm or engine_val_norm

    is_corrected = len(corrections) > 0
    return reconciled, is_corrected, corrections