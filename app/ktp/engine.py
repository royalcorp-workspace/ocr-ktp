import re
import os
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from app.ktp.extractor.common import LABELS_PATTERN
from app.core.logging_config import ktp_logger as logger


def score_ocr_text(raw_text: str, candidate_name: str = "", is_v1: bool = False) -> dict:
    """Skor kualitas raw OCR text secara terstruktur dengan validasi keabsahan NIK."""
    if not raw_text or not raw_text.strip():
        return {"score": 0, "has_nik": False, "has_valid_nik": False, "matched_primary": 0, "parsed_count": 0}

    score = 0

    has_header = any(re.search(r'\b' + hdr + r'\b', raw_text, re.IGNORECASE) for hdr in ["PROVINSI", "KABUPATEN", "KOTA"])
    if has_header:
        score += 20

    primary_labels = {"NIK", "NAMA", "ALAMAT", "TEMPAT_TGL_LAHIR"}
    matched_primary = 0
    for label, pattern in LABELS_PATTERN.items():
        if re.search(pattern, raw_text, re.IGNORECASE):
            if label in primary_labels:
                score += 10
                matched_primary += 1
            else:
                score += 5

    from app.ktp.extractor.identity import extract_nik
    from app.ktp.extractor.validators import validate_nik_structure, sync_nik_with_birthdate

    nik_cand = extract_nik(None, raw_text)
    has_nik = False
    has_valid_nik = False

    if nik_cand:
        has_nik = True
        if is_v1:
            from app.ktp.extractor import KTPExtractor
            ext_temp = KTPExtractor().extract(raw_text)
            synced = sync_nik_with_birthdate(nik_cand, ext_temp.tanggal_lahir, ext_temp.jenis_kelamin)
            if synced and validate_nik_structure(synced, raw_text):
                has_valid_nik = True
                score += 50
            elif validate_nik_structure(nik_cand, raw_text):
                has_valid_nik = True
                score += 50
            else:
                score += 15
        else:
            if validate_nik_structure(nik_cand, raw_text):
                has_valid_nik = True
                score += 50
            else:
                score += 15
    else:
        for line in raw_text.splitlines():
            for token in line.split():
                clean_t = re.sub(r'[^A-Za-z0-9OolI\|!SsZzBbGqUcLtys]', '', token)
                if len(clean_t) >= 14 and sum(1 for c in clean_t if c.isdigit()) >= 8:
                    has_nik = True
                    score += 10
                    break
            if has_nik:
                break

    if re.search(r'\b(ISLAM|KRISTEN|KATHOLIK|HINDU|BUDDHA|KAWIN|BELUM|CERAI|LAKI|PEREMPUAN|WNI)\b', raw_text, re.IGNORECASE):
        score += 15

    if (has_header or has_nik or matched_primary >= 2) and any(natural in candidate_name for natural in ["Pure Grayscale", "Soft CLAHE Grayscale"]):
        score += 15

    parsed_count = 0
    if is_v1:
        from app.ktp.extractor import KTPExtractor
        extractor = KTPExtractor()
        parsed = extractor.extract(raw_text)

        if parsed.nama and len(parsed.nama.strip()) >= 3 and not parsed.nama.replace(" ", "").isdigit():
            score += 15
            parsed_count += 1

        if parsed.tanggal_lahir or (parsed.tempat_lahir and len(parsed.tempat_lahir.strip()) >= 3):
            score += 15
            parsed_count += 1

        if (parsed.alamat and len(parsed.alamat.strip()) >= 5) or parsed.rt_rw:
            score += 15
            parsed_count += 1

        civil_fields = [parsed.agama, parsed.status_perkawinan, parsed.pekerjaan, parsed.kewarganegaraan, parsed.jenis_kelamin, parsed.golongan_darah]
        valid_civil = [f for f in civil_fields if f and str(f).strip()]
        score += len(valid_civil) * 5
        parsed_count += len(valid_civil)

    return {
        "score": score,
        "has_nik": has_nik,
        "has_valid_nik": has_valid_nik,
        "matched_primary": matched_primary,
        "parsed_count": parsed_count,
    }


def _run_single_candidate(candidate):
    """Worker function: jalankan pytesseract.image_to_data untuk 1 kandidat, return hasil + skor + word_conf_map."""
    if len(candidate) == 4:
        name, cand_img, psm_config, is_v1 = candidate
    else:
        name, cand_img, psm_config = candidate
        is_v1 = False

    data = pytesseract.image_to_data(cand_img, lang='ind+eng', config=psm_config, output_type=pytesseract.Output.DICT)
    
    lines = []
    current_line = []
    last_line_num = -1
    word_conf_map = {}

    n_boxes = len(data.get('text', []))
    for i in range(n_boxes):
        w_text = data['text'][i].strip()
        line_num = data['line_num'][i]
        try:
            w_conf = float(data['conf'][i])
        except (ValueError, TypeError):
            w_conf = -1.0

        if not w_text:
            continue

        if last_line_num != -1 and line_num != last_line_num:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []

        current_line.append(w_text)
        last_line_num = line_num

        if w_conf >= 0:
            word_conf_map[w_text.upper()] = w_conf

    if current_line:
        lines.append(" ".join(current_line))

    raw_text = "\n".join(lines)
    result = score_ocr_text(raw_text, candidate_name=name, is_v1=is_v1)
    return {
        "name": name,
        "raw_text": raw_text,
        "score": result["score"],
        "has_nik": result["has_nik"],
        "has_valid_nik": result["has_valid_nik"],
        "matched_primary": result["matched_primary"],
        "parsed_count": result.get("parsed_count", 0),
        "word_conf_map": word_conf_map,
    }


def _pick_overall_best(all_results: list):
    """Strict 2-Pass Priority Selection: Candidate with has_valid_nik == True ALWAYS wins over invalid ones.
       Among valid ones, prefer candidates with higher matched_primary count and full-page view over crops."""
    if not all_results:
        return None
    valid_nik_cands = [r for r in all_results if r.get("has_valid_nik")]
    if valid_nik_cands:
        return max(
            valid_nik_cands, 
            key=lambda x: (x["score"], x["matched_primary"], 0 if "crop" in x["name"].lower() else 1)
        )
    return max(
        all_results, 
        key=lambda x: (x["score"], x["matched_primary"], 0 if "crop" in x["name"].lower() else 1)
    )


def run_tier_candidates(candidates: list, tier_name: str = "Tier 1", is_v1: bool = False) -> tuple:
    if not candidates:
        return None, False, []

    # Capped at 4 worker threads to match container 4.0 CPU cores
    max_workers = min(len(candidates), 4)
    logger.info(f"[{tier_name}] Executing {len(candidates)} candidates in parallel (ThreadPoolExecutor max_workers={max_workers})...")

    task_args = [(c[0], c[1], c[2], is_v1) for c in candidates]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_run_single_candidate, task_args))

    for res in results:
        logger.info(
            f"[{tier_name}] EVALUATED CANDIDATE: name='{res['name']}', score={res['score']}, "
            f"has_nik={res['has_nik']}, has_valid_nik={res['has_valid_nik']}, matched_primary={res['matched_primary']}, parsed_count={res.get('parsed_count', 0)}"
        )

    best_cand = _pick_overall_best(results)

    meets_early_exit = False
    if best_cand:
        if is_v1:
            if (best_cand.get("has_nik") or best_cand.get("has_valid_nik")) and best_cand.get("parsed_count", 0) >= 2 and best_cand["score"] >= 70:
                meets_early_exit = True
            elif best_cand.get("parsed_count", 0) >= 3 and best_cand["score"] >= 70:
                meets_early_exit = True
        else:
            if best_cand.get("has_valid_nik") and best_cand["score"] >= 75:
                meets_early_exit = True
            elif best_cand["score"] >= 75 and best_cand["matched_primary"] >= 2:
                meets_early_exit = True

    return best_cand, meets_early_exit, results


def run_candidates_tiered(image, is_v1: bool = False) -> tuple:
    import time
    from app.ktp.preprocessing import build_tier1_candidates, build_tier2_candidates, build_tier3_candidates

    t_start = time.perf_counter()
    total_candidates_executed = 0
    all_tier_results = []

    # --- TIER 1 ---
    from app.ktp.preprocessing import build_tier1_candidates, build_tier2_candidates, build_tier3_candidates, crop_roi_candidates
    tier1_candidates = build_tier1_candidates(image)
    tier1_candidates.extend(crop_roi_candidates(image))
    total_candidates_executed += len(tier1_candidates)
    best_tier1, exit_tier1, results_t1 = run_tier_candidates(tier1_candidates, tier_name="Tier 1", is_v1=is_v1)
    all_tier_results.extend(results_t1)

    overall_best = _pick_overall_best(all_tier_results)

    if exit_tier1 and overall_best and (overall_best.get("matched_primary", 0) >= 2 or overall_best.get("parsed_count", 0) >= 2):
        logger.info(
            f"EARLY EXIT TRIGGERED AT TIER 1! "
            f"Winner: '{overall_best['name']}', Score: {overall_best['score']}, "
            f"Total Candidates Executed: {total_candidates_executed}"
        )
        return (
            overall_best["raw_text"],
            overall_best["name"],
            overall_best["score"],
            total_candidates_executed,
            overall_best.get("word_conf_map", {}),
            all_tier_results,
        )

    # --- TIMEOUT GUARD BEFORE TIER 2 ---
    elapsed_t1 = time.perf_counter() - t_start
    if elapsed_t1 >= 18.0:
        logger.warning(
            f"TIMEOUT GUARD TRIGGERED BEFORE TIER 2 ({elapsed_t1:.2f}s >= 18.0s)! "
            f"Skipping further tiers to prevent latency spike."
        )
        if overall_best:
            return (
                overall_best["raw_text"],
                overall_best["name"],
                overall_best["score"],
                total_candidates_executed,
                overall_best.get("word_conf_map", {}),
                all_tier_results,
            )
        return "", "", 0, total_candidates_executed, {}, all_tier_results

    # --- TIER 2 ---
    logger.info("Tier 1 score did not trigger early exit. Proceeding to Tier 2 (Deep Preprocessing)...")
    tier2_candidates = build_tier2_candidates(image)
    total_candidates_executed += len(tier2_candidates)
    best_tier2, exit_tier2, results_t2 = run_tier_candidates(tier2_candidates, tier_name="Tier 2", is_v1=is_v1)
    all_tier_results.extend(results_t2)

    overall_best = _pick_overall_best(all_tier_results)

    if exit_tier2 and overall_best:
        logger.info(
            f"EARLY EXIT TRIGGERED AT TIER 2! "
            f"Winner: '{overall_best['name']}', Score: {overall_best['score']}, "
            f"Total Candidates Executed: {total_candidates_executed}"
        )
        return (
            overall_best["raw_text"],
            overall_best["name"],
            overall_best["score"],
            total_candidates_executed,
            overall_best.get("word_conf_map", {}),
            all_tier_results,
        )

    # --- TIMEOUT GUARD BEFORE TIER 3 ---
    elapsed = time.perf_counter() - t_start
    if elapsed >= 18.0:
        logger.warning(
            f"TIMEOUT GUARD TRIGGERED BEFORE TIER 3 ({elapsed:.2f}s >= 18.0s)! "
            f"Skipping Tier 3 rotation fallback to prevent latency spike."
        )
        if overall_best:
            return (
                overall_best["raw_text"],
                overall_best["name"],
                overall_best["score"],
                total_candidates_executed,
                overall_best.get("word_conf_map", {}),
                all_tier_results,
            )
        return "", "", 0, total_candidates_executed, {}, all_tier_results

    # --- TIER 3 ---
    logger.info("Tier 1 & 2 scores did not trigger early exit. Proceeding to Tier 3 (Rotation Fallback)...")
    tier3_candidates = build_tier3_candidates(image)
    total_candidates_executed += len(tier3_candidates)
    best_tier3, _, results_t3 = run_tier_candidates(tier3_candidates, tier_name="Tier 3", is_v1=is_v1)
    all_tier_results.extend(results_t3)

    overall_best = _pick_overall_best(all_tier_results)

    logger.info(
        f"TIERED PIPELINE COMPLETED. Winner: '{overall_best['name'] if overall_best else 'None'}', "
        f"Score: {overall_best['score'] if overall_best else 0}, "
        f"Total Candidates Executed: {total_candidates_executed}"
    )

    if overall_best:
        return (
            overall_best["raw_text"],
            overall_best["name"],
            overall_best["score"],
            total_candidates_executed,
            overall_best.get("word_conf_map", {}),
            all_tier_results,
        )
    return "", "", 0, total_candidates_executed, {}, all_tier_results


def run_candidates(candidates: list) -> tuple:
    """Fallback kompatibilitas: menjalankan daftar kandidat secara paralel dalam 1 batch."""
    best_cand, _, _ = run_tier_candidates(candidates, tier_name="Fallback Batch")
    if best_cand:
        return best_cand["raw_text"], best_cand["name"], best_cand["score"]
    return "", "", 0