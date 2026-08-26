import time
import cv2
import numpy as np

from app.ktp.preprocessing import normalize_image_size, auto_orient_image
from app.ktp.engine import run_candidates_tiered
from app.ktp.extractor import KTPExtractor
from app.core.logging_config import ktp_logger as logger

from app.ktp.v1.roi_config import normalize_canvas_v1
from app.ktp.v1.roi_engine import extract_all_roi
from app.ktp.v1.fallback import merge_roi_and_fallback_extract
from app.ktp.v1.schemas_v1 import KTPOcrResponseV1
from app.ktp.v1.regional_normalizer import normalize_regional_text
from app.ktp.consensus import _vote_field

def process_ktp_image_v1(image_bytes: bytes) -> KTPOcrResponseV1:
    if not image_bytes:
        raise ValueError("Image bytes must not be empty.")

    t_start = time.perf_counter()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image. Invalid or corrupted image format.")

    # 1. Normalisasi Ukuran & Orientasi Dasar (Sama seperti Pipeline v0)
    normalized_base_image = normalize_image_size(image)
    oriented_image = auto_orient_image(normalized_base_image)
    
    # 2. General OCR (Baseline) - Multi-Candidate Server Voting
    t_gen_start = time.perf_counter()
    res_tuple = run_candidates_tiered(oriented_image, is_v1=True)
    if len(res_tuple) == 6:
        best_raw_text, _, best_score, _, best_word_conf_map, all_tier_results = res_tuple
    else:
        best_raw_text, _, best_score, _, best_word_conf_map = res_tuple
        all_tier_results = []
        
    extractor = KTPExtractor()
    general_parsed_data = extractor.extract(best_raw_text)

    field_specific_raw_text = {}
    field_specific_word_map = {}

    # Multi-Candidate Server Field Recovery & Voting across all 15 fields
    if all_tier_results:
        # Pre-extract all candidate structures once
        candidate_extractions = []
        for cand in all_tier_results:
            cand_raw = cand.get("raw_text", "")
            cand_score = float(cand.get("score", 0))
            if "ROI" in cand.get("name", ""):
                cand_score = max(cand_score, 50.0)
            if cand_raw and cand_score > 0:
                cand_parsed = extractor.extract(cand_raw)
                candidate_extractions.append({
                    "raw_text": cand_raw,
                    "score": cand_score,
                    "name": cand.get("name", "tesseract"),
                    "word_conf_map": cand.get("word_conf_map", {}),
                    "parsed": cand_parsed
                })

        field_names = [
            "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
            "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
            "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
        ]

        for field in field_names:
            tess_entries = []
            for cand_item in candidate_extractions:
                val = getattr(cand_item["parsed"], field, None)
                if val and str(val).strip():
                    tess_entries.append({
                        "value": str(val).strip(),
                        "confidence": cand_item["score"],
                        "candidate_name": cand_item["name"]
                    })

            if tess_entries:
                vote_res = _vote_field(field, None, 0.0, tess_entries)
                voted_val = vote_res.get("value")
                if voted_val:
                    if field in ["kelurahan_desa", "kecamatan", "tempat_lahir"]:
                        voted_val = normalize_regional_text(str(voted_val), field)
                    if field == "nama":
                        from app.ktp.extractor.identity import assess_name_quality
                        if not assess_name_quality(str(voted_val)):
                            voted_val = None
                    if voted_val:
                        setattr(general_parsed_data, field, voted_val)

                        # Find matching candidate for raw text & word conf map
                        for cand_item in candidate_extractions:
                            if cand_item["raw_text"] and voted_val in cand_item["raw_text"]:
                                field_specific_raw_text[field] = cand_item["raw_text"]
                                field_specific_word_map[field] = cand_item["word_conf_map"]
                                break

        # NIK Character-level position consensus refinement
        voted_nik = getattr(general_parsed_data, "nik", None)
        if voted_nik and len(voted_nik) == 16 and voted_nik.isdigit():
            all_raws = [c["raw_text"] for c in candidate_extractions if c.get("raw_text")]
            from app.ktp.extractor.validators import vote_nik_character_level
            tgl_l = getattr(general_parsed_data, "tanggal_lahir", None)
            j_kel = getattr(general_parsed_data, "jenis_kelamin", None)
            refined_nik = vote_nik_character_level(voted_nik, all_raws, tgl_l, j_kel)
            final_nik_val = refined_nik if (refined_nik and isinstance(refined_nik, str)) else voted_nik

            if final_nik_val:
                setattr(general_parsed_data, "nik", final_nik_val)
                for cand_item in candidate_extractions:
                    if cand_item["raw_text"] and final_nik_val in cand_item["raw_text"]:
                        field_specific_raw_text["nik"] = cand_item["raw_text"]
                        field_specific_word_map["nik"] = cand_item["word_conf_map"]
                        break

        # Fallback loop for any remaining empty fields
        for field in field_names:
            curr_val = getattr(general_parsed_data, field, None)
            if not curr_val or str(curr_val).strip() == "":
                for cand_item in candidate_extractions:
                    cand_val = getattr(cand_item["parsed"], field, None)
                    if cand_val and str(cand_val).strip() != "":
                        if field in ["kelurahan_desa", "kecamatan", "tempat_lahir"]:
                            cand_val = normalize_regional_text(str(cand_val), field)
                        if field == "nama":
                            from app.ktp.extractor.identity import assess_name_quality
                            if not assess_name_quality(str(cand_val)):
                                continue
                        setattr(general_parsed_data, field, cand_val)
                        field_specific_raw_text[field] = cand_item["raw_text"]
                        field_specific_word_map[field] = cand_item["word_conf_map"]
                        break

        # Prefer birthdate with realistic birth year (1930 - 2015) over recent years (>2015)
        if general_parsed_data.tanggal_lahir:
            try:
                y_val = int(general_parsed_data.tanggal_lahir.split("-")[-1])
                if y_val > 2015:
                    for cand in all_tier_results:
                        c_raw = cand.get("raw_text", "")
                        if c_raw:
                            c_parsed = extractor.extract(c_raw)
                            if c_parsed.tanggal_lahir:
                                c_y = int(c_parsed.tanggal_lahir.split("-")[-1])
                                if 1930 <= c_y <= 2015:
                                    general_parsed_data.tanggal_lahir = c_parsed.tanggal_lahir
                                    break
            except Exception:
                pass

        # Sanitize nama with assess_name_quality
        if general_parsed_data.nama:
            from app.ktp.extractor.identity import assess_name_quality
            if not assess_name_quality(str(general_parsed_data.nama)):
                general_parsed_data.nama = None

        # Post-merge NIK consistency validation (read-only evidence logging)
        if general_parsed_data.nik and general_parsed_data.tanggal_lahir:
            from app.ktp.extractor.validators import is_nik_consistent_with_birthdate
            is_consistent = is_nik_consistent_with_birthdate(
                general_parsed_data.nik,
                general_parsed_data.tanggal_lahir,
                general_parsed_data.jenis_kelamin
            )
            logger.info(f"[SERVICE_V1] NIK-DOB Consistency Check: {is_consistent} (NIK: {general_parsed_data.nik})")
    
    # 3. Normalisasi Kanvas V1 & ROI OCR
    normalized_image = normalize_canvas_v1(oriented_image)
    
    t_roi_start = time.perf_counter()
    roi_results = extract_all_roi(normalized_image)
    t_roi_end = time.perf_counter()
    
    logger.info(f"[ROI DEBUG] Full ROI Engine Results: {roi_results}")

    # 4. Fallback Merge
    merged_dict = merge_roi_and_fallback_extract(
        roi_results=roi_results, 
        general_parsed_data=general_parsed_data,
        general_base_score=best_score,
        general_raw_text=best_raw_text,
        general_word_conf_map=best_word_conf_map,
        field_specific_raw_text=field_specific_raw_text,
        field_specific_word_map=field_specific_word_map
    )
    
    response_data = KTPOcrResponseV1(**merged_dict)
    
    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    roi_ms = round((t_roi_end - t_roi_start) * 1000, 2)
    gen_ms = round((t_roi_start - t_gen_start) * 1000, 2)
    
    logger.info(f"[V1] PIPELINE TIMING - General: {gen_ms}ms, ROI: {roi_ms}ms, Total: {total_ms}ms")
    
    return response_data
