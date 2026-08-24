import re
import cv2
import pytesseract
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from app.ktp.v1.roi_config import ROI_CONFIG, WHITELISTS
from app.ktp.preprocessing import soft_unsharp_mask
from app.core.logging_config import ktp_logger as logger
from app.ktp.v1.regional_normalizer import fix_common_ocr_typos, normalize_regional_text


def _run_single_roi_ocr(field_name: str, roi_image: np.ndarray) -> dict:
    """
    Eksekusi OCR pada satu potongan ROI dengan enhancement khusus per-field.
    """
    whitelist = WHITELISTS.get(field_name, "")
    psm_mode = 6 if field_name in ["alamat", "nama"] else 7
    
    config_str = f"--psm {psm_mode}"
    if whitelist:
        config_str += f" -c tessedit_char_whitelist=\"{whitelist}\""

    # Image Enhancement for ROI
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
    
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    final_roi_image = soft_unsharp_mask(enhanced)
    
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_roi_image = cv2.morphologyEx(final_roi_image, cv2.MORPH_OPEN, kernel_morph)

    try:
        data = pytesseract.image_to_data(
            final_roi_image, 
            lang='ind+eng', 
            config=config_str, 
            output_type=pytesseract.Output.DICT
        )
        
        lines = []
        current_line = []
        last_line_num = -1
        word_conf_map = {}
        total_conf = 0.0
        word_count = 0

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
                total_conf += w_conf
                word_count += 1

        if current_line:
            lines.append(" ".join(current_line))

        raw_text = " ".join(lines).strip()
        avg_conf = (total_conf / word_count) if word_count > 0 else 0.0

        # Sanitasi awal dari typos dan spasi
        raw_text = fix_common_ocr_typos(raw_text)

        logger.info(f"[ROI DEBUG] Field: {field_name} | Raw OCR Text: '{raw_text}' | Avg Conf: {avg_conf}")

        from app.ktp.extractor import KTPExtractor
        extractor = KTPExtractor()
        parsed = extractor.extract(raw_text)
        final_value = getattr(parsed, field_name, "")
        
        if final_value is None or str(final_value).strip() == "":
            final_value = ""

        # Direct Parsing Fallback untuk ROI Text (karena ROI crop tidak selalu berisi label penuh)
        if not final_value and raw_text:
            text_clean = re.sub(r'^(NIK|NAMA|TEMPAT|TANGGAL|LAHIR|JENIS|KELAMIN|ALAMAT|RT/RW|KEL/DESA|KECAMATAN|AGAMA|STATUS|PERKAWINAN|PEKERJAAN|KEWARGANEGARAAN|BERLAKU|HINGGA)\s*[:\.]?\s*', '', raw_text, flags=re.IGNORECASE).strip()
            
            if field_name == "nik":
                digits = re.sub(r'\D', '', text_clean)
                if len(digits) == 16:
                    final_value = digits
            elif field_name == "rt_rw":
                match = re.search(r'(\d{3}\s*/\s*\d{3})', raw_text)
                if match:
                    final_value = match.group(1).replace(" ", "")
                else:
                    digits = re.sub(r'\D', '', raw_text)
                    if len(digits) == 6:
                        final_value = f"{digits[:3]}/{digits[3:]}"
            elif field_name == "tanggal_lahir":
                match = re.search(r'\d{2}-\d{2}-\d{4}', raw_text)
                if match:
                    final_value = match.group(0)
            elif field_name in ["kecamatan", "kelurahan_desa", "tempat_lahir"]:
                final_value = normalize_regional_text(text_clean, field_name)
            elif field_name == "alamat":
                final_value = text_clean
            elif field_name == "agama":
                if any(w in text_clean.upper() for w in ["ISLAM", "1SLAM", "SLAM"]):
                    final_value = "ISLAM"
                else:
                    m = re.search(r'\b(KRISTEN|KATHOLIK|KATOLIK|HINDU|BUDDHA|KHONGHUCU)\b', text_clean, re.IGNORECASE)
                    if m:
                        final_value = "KATHOLIK" if "KAT" in m.group(1).upper() else m.group(1).upper()
            elif field_name == "status_perkawinan":
                up = text_clean.upper()
                if "CERAI" in up:
                    final_value = "CERAI MATI" if "MATI" in up else "CERAI HIDUP"
                elif any(w in up for w in ["KAWIN", "KAW1N", "KAWLN"]):
                    final_value = "BELUM KAWIN" if any(b in up for b in ["BELUM", "8ELUM", "8ELUM"]) else "KAWIN"
                elif "BELUM" in up:
                    final_value = "BELUM KAWIN"
            elif field_name == "jenis_kelamin":
                up = text_clean.upper()
                if any(w in up for w in ["LAKI", "LAK1", "TAKILAKI", "MALE"]):
                    final_value = "LAKI-LAKI"
                elif any(w in up for w in ["PEREMPUAN", "PEREM", "FEMALE"]):
                    final_value = "PEREMPUAN"
            elif field_name == "pekerjaan":
                final_value = text_clean
            elif field_name == "kewarganegaraan":
                up = text_clean.upper()
                if any(w in up for w in ["WNI", "WN1", "W N I", "WINI"]):
                    final_value = "WNI"
                elif "WNA" in up:
                    final_value = "WNA"
            elif field_name == "berlaku_hingga":
                up = text_clean.upper()
                if any(w in up for w in ["SEUMUR", "HIDUP", "HUMOR", "HIDUR"]):
                    final_value = "SEUMUR HIDUP"
                else:
                    final_value = text_clean

        # Sanitasi khusus Nama (hapus leading quotes/dashes/numbers/noise)
        if field_name == "nama":
            raw_target = str(final_value) if final_value else raw_text
            # Split pada symbol pemisah seperti em-dash —, dash -, equal =, plus +
            part = re.split(r'[\u2010-\u2015\u2013\u2014\-_=\+]', raw_target)[0]
            words = [re.sub(r'[^A-Za-z]', '', w).upper() for w in part.split()]
            valid_words = [w for w in words if len(w) >= 2 and w not in ["NAMA", "NAME"]]
            if valid_words:
                final_value = " ".join(valid_words)

        # Normalisasi Wilayah jika belum terjamah
        if field_name in ["kecamatan", "kelurahan_desa", "tempat_lahir"] and final_value:
            final_value = normalize_regional_text(final_value, field_name)

        logger.info(f"[ROI DEBUG] Field: {field_name} | Extracted Value: '{final_value}'")

        return {
            "field": field_name,
            "raw_text": raw_text,
            "extracted_value": str(final_value).strip(),
            "confidence": avg_conf,
            "word_conf_map": word_conf_map
        }
    except Exception as e:
        logger.error(f"[ROI DEBUG] Exception in _run_single_roi_ocr for field {field_name}: {str(e)}")
        return {
            "field": field_name,
            "raw_text": "",
            "extracted_value": "",
            "confidence": 0.0,
            "word_conf_map": {}
        }


def extract_all_roi(normalized_image: np.ndarray) -> dict:
    h, w = normalized_image.shape[:2]
    roi_images = {}

    for field, coords in ROI_CONFIG.items():
        x_min = int(coords["x_min"] * w)
        y_min = int(coords["y_min"] * h)
        x_max = int(coords["x_max"] * w)
        y_max = int(coords["y_max"] * h)

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        roi_img = normalized_image[y_min:y_max, x_min:x_max]
        roi_images[field] = roi_img

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_single_roi_ocr, field, img): field 
            for field, img in roi_images.items()
        }
        for future in futures:
            res = future.result()
            results[res["field"]] = res

    return results
