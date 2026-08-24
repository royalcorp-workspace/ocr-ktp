"""
Weighted Consensus OCR Engine untuk KTP-el Indonesia.

Modul ini mengimplementasikan logika "Weighted Voting" yang menggabungkan
hasil OCR dari Mobile (Sistem A) dengan hasil OCR dari beberapa kandidat
preprocessing Tesseract (Sistem C) untuk menentukan nilai final per-field.
"""

import re
import time
import cv2
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from app.ktp.extractor import KTPExtractor
from app.ktp.preprocessing import (
    normalize_image_size,
    auto_orient_image,
    build_tier1_candidates,
    build_tier2_candidates,
)
from app.ktp.engine import _run_single_candidate, score_ocr_text
from app.core.logging_config import ktp_logger as logger


# ============================================================
# Field yang diikutsertakan dalam weighted voting.
# Tambahkan field baru cukup di sini + update MobileOCRInput schema.
# ============================================================
CONSENSUS_FIELDS = [
    "nik", "nama", "tempat_lahir", "tanggal_lahir", "golongan_darah",
    "kelurahan_desa", "kecamatan", "agama", "status_perkawinan",
    "pekerjaan", "kewarganegaraan", "berlaku_hingga"
]

# Field yang perlu pencocokan case-insensitive & normalisasi whitespace
CASE_INSENSITIVE_FIELDS = {"nama", "tempat_lahir", "kelurahan_desa", "kecamatan", "pekerjaan", "golongan_darah"}


# ============================================================
# Image Quality Analyzer (Laplacian Sharpness + Mean Brightness)
# ============================================================

def analyze_image_quality(image: np.ndarray) -> dict:
    """
    Menghitung metrik kualitas gambar KTP menggunakan OpenCV.

    Returns:
        dict berisi:
        - sharpness: Variance dari Laplacian (semakin tinggi = semakin tajam).
        - brightness: Rata-rata intensitas pixel (0-255).
        - score: Skor kualitas gabungan (0-100) berdasarkan sharpness & brightness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Sharpness: Laplacian variance (makin tinggi = makin fokus/tajam)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Brightness: Mean pixel intensity
    brightness = float(np.mean(gray))

    # Quality score heuristik:
    # - Sharpness ideal 50-500 (score 0-50)
    # - Brightness ideal 80-200 (score 0-50)
    sharpness_score = min(50.0, max(0.0, (min(sharpness, 500.0) / 500.0) * 50.0))

    if 80 <= brightness <= 200:
        brightness_score = 50.0
    elif brightness < 80:
        brightness_score = max(0.0, (brightness / 80.0) * 50.0)
    else:
        brightness_score = max(0.0, (1.0 - (brightness - 200.0) / 55.0) * 50.0)

    quality_score = round(sharpness_score + brightness_score, 1)

    return {
        "score": quality_score,
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
    }


# ============================================================
# Normalisasi string untuk pencocokan voting
# ============================================================

def _normalize_for_match(value: str, field: str) -> str:
    """
    Normalisasi string untuk digunakan sebagai key voting.
    - Hapus spasi berlebih.
    - Case-insensitive untuk field nama/tempat_lahir -> uppercase.
    - Untuk NIK/tanggal, pertahankan apa adanya (hanya strip).
    """
    if not value:
        return ""
    normalized = " ".join(value.split())  # collapse whitespace
    if field in CASE_INSENSITIVE_FIELDS:
        normalized = normalized.upper()
    return normalized


# ============================================================
# Weighted Voting Logic
# ============================================================

def _vote_field(
    field: str,
    mobile_value: str | None,
    mobile_confidence: float,
    tesseract_entries: list[dict],
) -> dict:
    """
    Lakukan weighted voting untuk satu field.

    1. Total skor tertinggi (akumulasi) digunakan HANYA untuk menentukan string pemenang (`winning_value`).
    2. Nilai `confidence` final dihitung dari RATA-RATA (Average) skor confidence dari para pemilih
       (Mobile maupun Tesseract) yang memilih string pemenang tersebut.
    3. Capping nilai final `confidence` maksimal 100.0 (skala 0.0 - 100.0%).

    Args:
        field: Nama field (misal "nik", "nama").
        mobile_value: Nilai string dari Mobile.
        mobile_confidence: Confidence score dari Mobile (0-100).
        tesseract_entries: List of {"value": str, "confidence": float, "candidate_name": str}.

    Returns:
        dict: {"value": str, "confidence": float, "source": str, "validated": bool}
    """
    # Akumulasi skor per unique normalized string (untuk penentuan pemenang/voting)
    vote_scores: dict[str, float] = defaultdict(float)
    # Kumpulkan skor confidence pemilih per normalized string (untuk perhitungan rata-rata)
    vote_voter_scores: dict[str, list[float]] = defaultdict(list)
    vote_sources: dict[str, list[str]] = defaultdict(list)
    # Simpan original (non-normalized) value terbaik per key
    vote_originals: dict[str, str] = {}
    vote_max_conf: dict[str, float] = {}

    from app.ktp.v1.fallback import _sanity_check_free_text

    # --- 1. Kumpulkan Suara Mobile ---
    mob_key = None
    mob_conf = 0.0
    is_mob_strong = False

    if mobile_value and mobile_value.strip() and mobile_value.strip() != "-":
        is_mob_sane, _ = _sanity_check_free_text(field, mobile_value)
        norm_key = _normalize_for_match(mobile_value, field)
        if norm_key and is_mob_sane:
            mob_key = norm_key
            mob_conf = float(mobile_confidence)
            if mob_conf >= 75.0:
                is_mob_strong = True
            voter_conf = min(100.0, max(0.0, mob_conf))
            vote_scores[norm_key] += mob_conf
            vote_voter_scores[norm_key].append(voter_conf)
            vote_sources[norm_key].append("mobile")
            vote_originals[norm_key] = mobile_value.strip()
            vote_max_conf[norm_key] = mob_conf

    # --- 2. Kumpulkan Suara Tesseract dengan Sanity Gate ---
    for entry in tesseract_entries:
        val = entry.get("value")
        conf = float(entry.get("confidence", 0.0))
        cand_name = entry.get("candidate_name", "tesseract")
        if val and val.strip():
            is_cand_sane, _ = _sanity_check_free_text(field, val)
            if not is_cand_sane:
                continue  # SANITY GATE: Skip garbled/noise Tesseract candidates!

            norm_key = _normalize_for_match(val, field)
            if norm_key:
                voter_conf = min(100.0, max(0.0, conf))

                # General Regional & Format Consistency Bonus for NIK (Provinsi 11-92):
                if field == "nik" and len(norm_key) == 16 and norm_key.isdigit():
                    from app.ktp.extractor.validators import PROVINCE_CODES
                    if any(p_code == norm_key[:2] for p_code in PROVINCE_CODES.values()):
                        voter_conf += 10.0
                        conf += 10.0

                vote_scores[norm_key] += conf
                vote_voter_scores[norm_key].append(voter_conf)
                vote_sources[norm_key].append(f"tesseract:{cand_name}")
                if norm_key not in vote_originals or conf > vote_max_conf.get(norm_key, -1.0):
                    vote_originals[norm_key] = val.strip()
                    vote_max_conf[norm_key] = conf

    if not vote_scores:
        return {
            "value": None,
            "confidence": 0.0,
            "source": "none",
            "validated": False,
        }

    # --- 2.5 Filter Tesseract Candidates against Mobile Priority Bonus & Override Gate ---
    if is_mob_strong and mob_key in vote_scores:
        filtered_scores = {}
        for key in vote_scores:
            if key == mob_key:
                filtered_scores[key] = mob_conf
            else:
                voters = vote_voter_scores[key]
                cand_max_conf = vote_max_conf.get(key, 0.0)
                cand_voter_count = len(voters)
                cand_avg_conf = (sum(voters) / cand_voter_count) if cand_voter_count > 0 else 0.0
                cand_effective_score = cand_max_conf if cand_voter_count == 1 else cand_avg_conf

                # Tesseract Override Gate:
                # 1. Single candidate max conf >= 98.0, OR
                # 2. Konsensus Mutlak Tesseract: N >= 2 candidates AND average conf >= 85.0
                #    (P2.2: kandidat tsb juga harus lulus sanity check)
                is_single_strong = cand_max_conf >= 98.0
                _cand_orig = vote_originals.get(key, "")
                _cand_sane, _ = _sanity_check_free_text(field, _cand_orig) if _cand_orig else (False, "empty")
                is_absolute_consensus = (cand_voter_count >= 2 and cand_avg_conf >= 85.0 and _cand_sane)

                if is_single_strong or is_absolute_consensus:
                    filtered_scores[key] = cand_effective_score
                else:
                    # Cap Tesseract candidate score below mobile_data so garbled accumulation cannot win
                    filtered_scores[key] = min(cand_effective_score, mob_conf - 5.0)
        target_scores = filtered_scores
    else:
        target_scores = vote_scores

    # --- 3. Minimum Quality Gate ---
    # Jika max conf semua kandidat < 50.0, tidak ada kandidat yang cukup yakin
    # Return None agar tidak menghasilkan nilai salah yang terlihat plausible
    max_any_conf = max(target_scores.values()) if target_scores else 0.0
    if max_any_conf < 50.0 and not is_mob_strong:
        return {
            "value": None,
            "confidence": 0.0,
            "source": "none",
            "validated": False,
        }

    # --- 4. Tentukan STRING PEMENANG dengan Tie-Breaker Priority ---
    winner_key = max(target_scores, key=target_scores.get)

    if is_mob_strong and mob_key in target_scores and winner_key != mob_key:
        score_diff = target_scores[winner_key] - target_scores[mob_key]
        if score_diff <= 10.0:
            winner_key = mob_key  # TIE-BREAKER: mobile_data wins ties or <= 10.0 diff!

    winning_value = vote_originals[winner_key]
    if field == "alamat" and winning_value:
        from app.ktp.extractor.address import extract_alamat
        cleaned_addr = extract_alamat(winning_value)
        if cleaned_addr:
            winning_value = cleaned_addr

    winner_sources = vote_sources[winner_key]
    skor_pemilih_pemenang = vote_voter_scores[winner_key]

    # --- 4. Hitung RATA-RATA confidence pemilih pemenang & lakukan capping max 100.0 ---
    if skor_pemilih_pemenang:
        avg_confidence = sum(skor_pemilih_pemenang) / len(skor_pemilih_pemenang)
    else:
        avg_confidence = 0.0

    final_confidence = round(min(100.0, max(0.0, avg_confidence)), 1)

    # Build source label
    has_mobile = any(s == "mobile" for s in winner_sources)
    has_tesseract = any(s.startswith("tesseract:") for s in winner_sources)
    if has_mobile and has_tesseract:
        source_label = "mobile+tesseract"
    elif has_mobile:
        source_label = "mobile"
    elif has_tesseract:
        source_label = "tesseract"
    else:
        source_label = "none"

    # Validated: true jika didukung oleh >= 2 sumber ATAU final_confidence >= 80.0
    voter_count = len(winner_sources)
    validated = voter_count >= 2 or final_confidence >= 80.0

    return {
        "value": winning_value,
        "confidence": final_confidence,
        "source": source_label,
        "validated": validated,
        "voter_count": voter_count,
        "raw_sources": winner_sources,
    }


# ============================================================
# Main Consensus OCR Pipeline
# ============================================================

def run_consensus_ocr(image_bytes: bytes, mobile_data: dict) -> dict:
    """
    Pipeline utama Weighted Consensus OCR.

    1. Decode & normalize image.
    2. Hitung image quality metrics.
    3. Jalankan Tier 1 candidates secara paralel, kumpulkan SEMUA hasil per-kandidat.
    4. Jika Tier 1 tidak cukup (tidak ada NIK terbaca), lanjut ke Tier 2.
    5. Ekstrak field terstruktur dari setiap raw_text kandidat.
    6. Lakukan weighted voting per-field antara data Mobile + semua kandidat Tesseract.
    7. Return hasil konsensus beserta quality metrics.

    Args:
        image_bytes: Raw bytes gambar KTP.
        mobile_data: Dict hasil parse dari MobileOCRInput.model_dump().

    Returns:
        dict: {
            "success": bool,
            "data": { field: ValidatedField_dict, ... },
            "quality": { "score": ..., "sharpness": ..., "brightness": ... },
            "warnings": [str, ...],
        }
    """
    t_start = time.perf_counter()
    warnings = []

    # --- 1. Decode & Normalize Image ---
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "success": False,
            "data": {},
            "quality": {"score": 0, "sharpness": 0, "brightness": 0},
            "warnings": ["Gagal decode gambar. Format tidak valid atau file corrupt."],
        }

    image = normalize_image_size(image)
    image = auto_orient_image(image)

    # --- 2. Image Quality ---
    quality = analyze_image_quality(image)
    if quality["sharpness"] < 30:
        warnings.append("Gambar terdeteksi blur/tidak fokus (sharpness rendah).")
    if quality["brightness"] < 60:
        warnings.append("Gambar terlalu gelap (brightness rendah).")
    elif quality["brightness"] > 220:
        warnings.append("Gambar terlalu terang/overexposed (brightness tinggi).")

    # --- 3. Jalankan Tier 1 kandidat secara PARALEL, kumpulkan SEMUA hasil ---
    from app.ktp.preprocessing import crop_roi_candidates
    tier1_candidates = build_tier1_candidates(image)
    tier1_candidates.extend(crop_roi_candidates(image))
    all_ocr_results = []

    max_workers = len(tier1_candidates)
    logger.info(f"[Consensus] Running Tier 1 ({len(tier1_candidates)} candidates) in parallel...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tier1_results = list(executor.map(_run_single_candidate, tier1_candidates))
    all_ocr_results.extend(tier1_results)

    # Cek apakah ada NIK terbaca di Tier 1
    any_nik_tier1 = any(r.get("has_nik", False) for r in tier1_results)
    best_score_tier1 = max((r.get("score", 0) for r in tier1_results), default=0)

    # --- 4. Tier 2 fallback jika Tier 1 kurang memadai ---
    if not any_nik_tier1 and best_score_tier1 < 40:
        logger.info("[Consensus] Tier 1 insufficient. Running Tier 2 fallback...")
        tier2_candidates = build_tier2_candidates(image)
        with ThreadPoolExecutor(max_workers=len(tier2_candidates)) as executor:
            tier2_results = list(executor.map(_run_single_candidate, tier2_candidates))
        all_ocr_results.extend(tier2_results)

    total_candidates = len(all_ocr_results)
    logger.info(f"[Consensus] Total candidates executed: {total_candidates}")

    # --- 5. Ekstrak field dari setiap raw_text kandidat ---
    extractor = KTPExtractor()
    candidate_extractions = []

    for res in all_ocr_results:
        raw_text = res.get("raw_text", "")
        candidate_score = res.get("score", 0)
        candidate_name = res.get("name", "unknown")

        if not raw_text.strip() or candidate_score <= 0:
            continue

        parsed = extractor.extract(raw_text)
        parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)

        candidate_extractions.append({
            "parsed": parsed_dict,
            "score": candidate_score,
            "name": candidate_name,
        })

    logger.info(f"[Consensus] Candidates with extractable data: {len(candidate_extractions)}")

    # --- 6. Weighted Voting per-field ---
    consensus_data = {}

    for field in CONSENSUS_FIELDS:
        # Data Mobile
        mobile_field_data = mobile_data.get(field)
        if mobile_field_data and isinstance(mobile_field_data, dict):
            mob_value = mobile_field_data.get("value")
            mob_confidence = mobile_field_data.get("confidence", 0.0)
        else:
            mob_value = None
            mob_confidence = 0.0

        # Data Tesseract (kumpulkan dari semua kandidat)
        tess_entries = []
        for cand in candidate_extractions:
            cand_value = cand["parsed"].get(field)
            if cand_value and str(cand_value).strip():
                # Gunakan score kandidat sebagai confidence Tesseract
                tess_entries.append({
                    "value": str(cand_value).strip(),
                    "confidence": float(cand["score"]),
                    "candidate_name": cand["name"],
                })

        # Voting
        result = _vote_field(field, mob_value, mob_confidence, tess_entries)
        consensus_data[field] = result

    # --- 6.5. Post-Processing & Calibrated Scoring ---
    # 6.5.a. NIK Character-Level Voting
    preliminary_data = {f: (consensus_data[f]["value"] or "") for f in CONSENSUS_FIELDS}
    winning_nik = preliminary_data.get("nik")
    if winning_nik and len(winning_nik) == 16:
        from app.ktp.extractor.validators import vote_nik_character_level
        raw_texts = [res.get("raw_text", "") for res in all_ocr_results]
        refined_nik = vote_nik_character_level(
            base_nik=winning_nik,
            raw_texts=raw_texts,
            tanggal_lahir=preliminary_data.get("tanggal_lahir"),
            jenis_kelamin=None  # We don't extract gender in validate yet, but DOB is enough for some syncs
        )
        if refined_nik != winning_nik:
            logger.info(f"[Consensus] NIK Character-Level Voting applied: '{winning_nik}' -> '{refined_nik}'")
            consensus_data["nik"]["value"] = refined_nik
            preliminary_data["nik"] = refined_nik

    # 6.5.b. Calibrated Gated Confidence
    from app.ktp.confidence import calculate_field_confidence
    best_cand = max(all_ocr_results, key=lambda x: x.get("score", 0)) if all_ocr_results else {}
    best_raw_text = best_cand.get("raw_text", "")
    best_word_conf_map = best_cand.get("word_conf_map", {})

    for field in CONSENSUS_FIELDS:
        val = consensus_data[field]["value"]
        if val:
            # Use the average confidence from all winning voters as the base score
            base_conf = int(consensus_data[field]["confidence"])
            calc_conf = calculate_field_confidence(
                field_name=field,
                value=val,
                base_score=base_conf,
                word_conf_map=best_word_conf_map,
                raw_text=best_raw_text,
                all_fields=preliminary_data
            )
            consensus_data[field]["confidence"] = calc_conf
            voter_count = consensus_data[field]["voter_count"]
            consensus_data[field]["validated"] = voter_count >= 2 or calc_conf >= 80.0
            
            # clean up voter_count from output if needed, or leave it
            consensus_data[field].pop("voter_count", None)

        logger.info(
            f"[Consensus] VOTE field='{field}': "
            f"winner='{consensus_data[field]['value']}' confidence={consensus_data[field]['confidence']} "
            f"source={consensus_data[field]['source']} validated={consensus_data[field]['validated']}"
        )

    # --- 7. Final result ---
    t_end = time.perf_counter()
    duration_ms = round((t_end - t_start) * 1000, 2)

    logger.info(
        f"[Consensus] Pipeline completed in {duration_ms}ms. "
        f"Candidates: {total_candidates}, Warnings: {len(warnings)}"
    )

    return {
        "success": True,
        "data": consensus_data,
        "quality": quality,
        "warnings": warnings,
    }
