import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from app.ktp.consensus import analyze_image_quality, _vote_field
from app.ktp.engine import _run_single_candidate
from app.ktp.preprocessing import build_tier1_candidates, build_tier2_candidates, crop_roi_candidates, auto_orient_image
from app.ktp.extractor import KTPExtractor
from app.core.logging_config import ktp_logger as logger
from app.ktp.confidence import calculate_field_confidence

from app.ktp.v1.roi_config import normalize_canvas_v1
from app.ktp.v1.roi_engine import extract_all_roi
from app.ktp.v1.schemas_v1 import ConsensusResponseV1, KTPOcrResponseV1

CONSENSUS_FIELDS_V1 = [
    "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
    "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
    "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
]

def run_consensus_ocr_v1(image_bytes: bytes, mobile_data: dict) -> dict:
    t_start = time.perf_counter()
    warnings = []

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "data": {}, "quality": {"score": 0, "sharpness": 0, "brightness": 0}, "warnings": ["Format gambar tidak valid."]}

    # Orientasi dan Cek Kualitas
    oriented_image = auto_orient_image(image)
    quality = analyze_image_quality(oriented_image)

    # 1. GENERAL OCR (Tier 1 & 2)
    tier1_candidates = build_tier1_candidates(oriented_image)
    tier1_candidates.extend(crop_roi_candidates(oriented_image))
    all_ocr_results = []

    with ThreadPoolExecutor(max_workers=len(tier1_candidates)) as executor:
        tier1_results = list(executor.map(_run_single_candidate, tier1_candidates))
    all_ocr_results.extend(tier1_results)

    any_nik_tier1 = any(r.get("has_nik", False) for r in tier1_results)
    best_score_tier1 = max((r.get("score", 0) for r in tier1_results), default=0)

    if not any_nik_tier1 and best_score_tier1 < 40:
        tier2_candidates = build_tier2_candidates(oriented_image)
        with ThreadPoolExecutor(max_workers=len(tier2_candidates)) as executor:
            tier2_results = list(executor.map(_run_single_candidate, tier2_candidates))
        all_ocr_results.extend(tier2_results)

    # Ekstrak data dari semua kandidat General OCR
    extractor = KTPExtractor()
    candidate_extractions = []
    for res in all_ocr_results:
        raw_text = res.get("raw_text", "")
        if not raw_text.strip() or res.get("score", 0) <= 0:
            continue
        parsed = extractor.extract(raw_text)
        candidate_extractions.append({
            "parsed": parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed),
            "score": res["score"],
            "name": res["name"],
        })

    # 2. WEIGHTED CONSENSUS (General OCR vs Mobile Data)
    consensus_general = {}
    preliminary_data = {}
    
    for field in CONSENSUS_FIELDS_V1:
        mob_field = mobile_data.get(field)
        mob_val = mob_field.get("value") if mob_field else None
        mob_conf = mob_field.get("confidence", 0.0) if mob_field else 0.0

        tess_entries = []
        for cand in candidate_extractions:
            cand_val = cand["parsed"].get(field)
            if cand_val and str(cand_val).strip():
                cand_conf = min(95.0, float(cand["score"])) if float(cand["score"]) > 100.0 else float(cand["score"])
                tess_entries.append({
                    "value": str(cand_val).strip(),
                    "confidence": cand_conf,
                    "candidate_name": cand["name"]
                })
        
        vote_res = _vote_field(field, mob_val, mob_conf, tess_entries)
        consensus_general[field] = vote_res
        preliminary_data[field] = vote_res.get("value") or ""

    # NIK Post-Voting Bi-Directional Cross Validation (Refinement untuk Kategori B)
    voted_nik = consensus_general.get("nik", {}).get("value")
    voted_dob = consensus_general.get("tanggal_lahir", {}).get("value")
    voted_jk = consensus_general.get("jenis_kelamin", {}).get("value")
    if voted_nik and len(voted_nik) == 16 and voted_nik.isdigit():
        from app.ktp.extractor.validators import is_nik_consistent_with_birthdate, sync_nik_with_birthdate, vote_nik_character_level, validate_nik_structure
        if not is_nik_consistent_with_birthdate(voted_nik, voted_dob, voted_jk):
            all_raws = [c.get("raw_text", "") for c in all_ocr_results if c.get("raw_text")]
            refined_nik = vote_nik_character_level(voted_nik, all_raws, voted_dob, voted_jk)
            if refined_nik and validate_nik_structure(refined_nik):
                consensus_general["nik"]["value"] = refined_nik
                preliminary_data["nik"] = refined_nik

    # Kalibrasi Consensus Confidence menggunakan Tesseract Word Map dari kandidat spesifik pemenang
    best_cand = max(all_ocr_results, key=lambda x: x.get("score", 0)) if all_ocr_results else {}
    fallback_raw_text = best_cand.get("raw_text", "")
    fallback_word_conf_map = best_cand.get("word_conf_map", {})

    for field in CONSENSUS_FIELDS_V1:
        val = consensus_general[field]["value"]
        if val:
            base_conf = int(consensus_general[field]["confidence"])
            raw_sources = consensus_general[field].get("raw_sources", [])
            
            specific_raw_text = fallback_raw_text
            specific_word_conf_map = fallback_word_conf_map
            
            # Cari kandidat tesseract pemenang spesifik jika ada
            for s in raw_sources:
                s_str = str(s)
                if s_str.startswith("tesseract:"):
                    cand_name = s_str.split("tesseract:")[1]
                    for cand in all_ocr_results:
                        if cand.get("name") == cand_name:
                            specific_raw_text = cand.get("raw_text", "")
                            specific_word_conf_map = cand.get("word_conf_map", {})
                            break
                    break
                    
            calc_conf = calculate_field_confidence(
                field_name=field,
                value=val,
                base_score=base_conf,
                word_conf_map=specific_word_conf_map,
                raw_text=specific_raw_text,
                all_fields=preliminary_data
            )
            consensus_general[field]["confidence"] = calc_conf
            # Cleanup raw_sources from dict since it's not in the response schema
            consensus_general[field].pop("raw_sources", None)

    # 3. ROI OCR
    normalized_image = normalize_canvas_v1(oriented_image)
    roi_results = extract_all_roi(normalized_image)

    logger.info(f"[ROI DEBUG] Full ROI Engine Results (Validate): {roi_results}")

    # 4. FALLBACK MERGE (ROI vs Consensus General)
    from app.ktp.v1.fallback import merge_roi_and_fallback_validate
    final_merged = merge_roi_and_fallback_validate(roi_results, consensus_general)

    response_data = KTPOcrResponseV1(**final_merged)
    return {
        "success": True,
        "data": response_data.model_dump(),
        "quality": quality,
        "warnings": warnings,
    }
