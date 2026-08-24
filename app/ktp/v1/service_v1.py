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

    # Multi-Candidate Server Field Recovery & NIK Consensus Voting
    if all_tier_results:
        # NIK Voting across all server candidates (including ROI NIK crop)
        nik_tess_entries = []
        for cand in all_tier_results:
            cand_raw = cand.get("raw_text", "")
            if cand_raw and (cand.get("score", 0) > 0 or "ROI" in cand.get("name", "")):
                cand_parsed = extractor.extract(cand_raw)
                cand_nik = getattr(cand_parsed, "nik", None)
                if cand_nik and str(cand_nik).strip():
                    # Memberikan boost confidence 100 untuk ROI NIK
                    cand_score = 100.0 if "ROI" in cand.get("name", "") else float(cand.get("score", 0))
                    nik_tess_entries.append({
                        "value": str(cand_nik).strip(),
                        "confidence": cand_score,
                        "candidate_name": cand.get("name", "tesseract")
                    })

        if nik_tess_entries:
            nik_vote_res = _vote_field("nik", None, 0.0, nik_tess_entries)
            voted_nik = nik_vote_res.get("value")
            
            # Character-level position consensus refinement across all candidate raw_texts
            all_raws = [c.get("raw_text", "") for c in all_tier_results if c.get("raw_text")]
            from app.ktp.extractor.validators import vote_nik_character_level
            tgl_l = getattr(general_parsed_data, "tanggal_lahir", None)
            j_kel = getattr(general_parsed_data, "jenis_kelamin", None)
            voted_nik = vote_nik_character_level(voted_nik, all_raws, tgl_l, j_kel)

            if voted_nik:
                setattr(general_parsed_data, "nik", voted_nik)
                for cand in all_tier_results:
                    cand_raw = cand.get("raw_text", "")
                    if cand_raw and voted_nik in cand_raw:
                        field_specific_raw_text["nik"] = cand_raw
                        field_specific_word_map["nik"] = cand.get("word_conf_map", {})
                        break
        field_names = [
            "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
            "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
            "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
        ]
        for field in field_names:
            curr_val = getattr(general_parsed_data, field, None)
            if curr_val and field in ["kelurahan_desa", "kecamatan", "tempat_lahir"]:
                curr_val = normalize_regional_text(str(curr_val), field)
                setattr(general_parsed_data, field, curr_val)

            if curr_val and str(curr_val).strip() != "":
                field_specific_raw_text[field] = best_raw_text
                field_specific_word_map[field] = best_word_conf_map
            else:
                for cand in all_tier_results:
                    cand_raw = cand.get("raw_text", "")
                    if cand_raw and (cand.get("score", 0) > 0 or "ROI" in cand.get("name", "")):
                        cand_parsed = extractor.extract(cand_raw)
                        cand_val = getattr(cand_parsed, field, None)
                        if cand_val and field in ["kelurahan_desa", "kecamatan", "tempat_lahir"]:
                            cand_val = normalize_regional_text(str(cand_val), field)
                        if cand_val and str(cand_val).strip() != "":
                            if field == "nama":
                                from app.ktp.extractor.identity import assess_name_quality
                                if not assess_name_quality(str(cand_val)):
                                    continue
                            cur_v = getattr(general_parsed_data, field, None)
                            if not cur_v or str(cur_v).strip() == "":
                                setattr(general_parsed_data, field, cand_val)
                                field_specific_raw_text[field] = cand_raw
                                field_specific_word_map[field] = cand.get("word_conf_map", {})
                                break
                            elif field == "nama" and len(str(cur_v).strip().split()) == 1 and len(str(cand_val).strip().split()) >= 2:
                                setattr(general_parsed_data, field, cand_val)
                                field_specific_raw_text[field] = cand_raw
                                field_specific_word_map[field] = cand.get("word_conf_map", {})
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

        # Post-merge NIK birthdate sync
        if general_parsed_data.nik and general_parsed_data.tanggal_lahir:
            from app.ktp.extractor.validators import sync_nik_with_birthdate
            synced_nik = sync_nik_with_birthdate(
                general_parsed_data.nik,
                general_parsed_data.tanggal_lahir,
                general_parsed_data.jenis_kelamin
            )
            if synced_nik:
                general_parsed_data.nik = synced_nik
    
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
