from app.ktp.v1.schemas_v1 import FieldWithSource
from app.ktp.confidence import calculate_field_confidence
from app.core.logging_config import ktp_logger as logger
import re

INVALID_LABELS_ROI = [
    "NIK", "NAMA", "TEMPAT", "TANGGAL", "LAHIR", "JENIS", "KELAMIN", 
    "GOL", "DARAH", "ALAMAT", "RT", "RW", "KEL", "DESA", "KECAMATAN",
    "AGAMA", "STATUS", "PERKAWINAN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU", "HINGGA"
]

def _sanity_check_free_text(field_name: str, text: str) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    if len(text.strip()) <= 1 and not (field_name == "golongan_darah" and text.strip().upper() in ["A", "B", "O"]):
        return False, "too short"
        
    text_upper = text.upper().strip()
    
    # 1. Field-Specific Strict Validation (domain whitelist)
    if field_name == "kewarganegaraan":
        if text_upper not in ["WNI", "WNA", "CHINA"]:
            return False, "invalid kewarganegaraan value"
            
    if field_name == "golongan_darah":
        if text_upper not in ["A", "B", "AB", "O", "-"]:
            return False, "invalid golongan_darah value"
            
    if field_name == "berlaku_hingga":
        if text_upper != "SEUMUR HIDUP" and not re.search(r'\d{2}-\d{2}-\d{4}', text_upper):
            return False, "invalid berlaku_hingga value"

    if field_name == "jenis_kelamin":
        if text_upper not in ["LAKI-LAKI", "PEREMPUAN"]:
            return False, "invalid jenis_kelamin value"

    if field_name == "agama":
        valid_agama = ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KHONGHUCU", "KEPERCAYAAN"]
        if text_upper not in valid_agama:
            return False, f"invalid agama value: '{text_upper}'"

    if field_name == "status_perkawinan":
        valid_status = ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"]
        if text_upper not in valid_status:
            return False, f"invalid status_perkawinan value: '{text_upper}'"

    if field_name == "nik":
        cleaned_nik = re.sub(r'\D', '', text_upper)
        from app.ktp.extractor.validators import validate_nik_structure
        if not validate_nik_structure(cleaned_nik):
            return False, f"invalid NIK structure: '{cleaned_nik}'"

    if field_name == "rt_rw":
        match = re.search(r'^\s*(\d{1,3})\s*/\s*(\d{1,3})\s*$', text_upper)
        if not match:
            return False, f"invalid RT/RW format: '{text_upper}'"

    # 2. Check for repetitive noise characters (dashes, equals, underscores, dots)
    if re.search(r'[-=_\u2014\u2013.]{2,}', text):
        return False, "contains repetitive noise symbols"

    # 3. Comprehensive Label Leakage Detection (mencegah teks field tetangga masuk ke candidate)
    leak_labels = [
        "NIK", "NAMA", "LAHIR", "KELAMIN", "DARAH", "ALAMAT",
        "KECAMATAN", "AGAMA", "PERKAWINAN", "PEKERJAAN", 
        "KEWARGANEGARAAN", "BERLAKU", "HINGGA", "SEUMUR", "HIDUP",
        "HARIAN", "PROVINSI", "KABUPATEN", "ALAMA", "KELDESA", "KELURAHAN",
        "DESA", "TEMPAT", "TEMPAL", "TEMPAIT", "TEMPALITGL", "TEMPAITGL", "TTL"
    ]
    # Setiap field boleh mengandung label dirinya sendiri
    exclude = []
    if field_name == "kewarganegaraan": exclude = ["KEWARGANEGARAAN"]
    elif field_name == "agama": exclude = ["AGAMA"]
    elif field_name == "status_perkawinan": exclude = ["PERKAWINAN"]
    elif field_name == "pekerjaan": exclude = ["PEKERJAAN", "HARIAN"]
    elif field_name == "berlaku_hingga": exclude = ["BERLAKU", "HINGGA", "SEUMUR", "HIDUP"]
    elif field_name == "golongan_darah": exclude = ["DARAH"]
    elif field_name == "jenis_kelamin": exclude = ["KELAMIN"]
    elif field_name == "tempat_lahir": exclude = ["LAHIR", "TEMPAT", "TEMPAL", "TEMPAIT", "TTL"]
    elif field_name == "kelurahan_desa": exclude = ["KELDESA", "KELURAHAN", "DESA"]
    elif field_name == "kecamatan": exclude = ["KECAMATAN"]
    elif field_name == "nama": exclude = ["NAMA"]
    elif field_name == "nik": exclude = ["NIK"]
    elif field_name == "alamat": exclude = ["ALAMAT", "ALAMA", "RT", "RW", "RT/RW", "RTRW", "RTIRW", "RT/AW", "RT/RAW", "RT/RN", "AT/AW", "AT/RW", "ATRW"]
    
    for label in leak_labels:
        if label not in exclude:
            if re.search(r'\b' + re.escape(label) + r'\b', text_upper) or (len(label) >= 4 and label in text_upper):
                return False, f"contains leaked label '{label}'"
                
    # 4. Symbol Ratio (excluding structural symbols for date, rt_rw, nik, and alamat)
    import string
    noise_chars = set(string.punctuation + "\u2014\u2013_|=+\u201c\u201d\u2014\u00ab\u00bb\u00b0\u00a9$")
    if field_name in ["tanggal_lahir", "berlaku_hingga", "rt_rw", "nik"]:
        symbols = sum(1 for c in text if c in noise_chars and c not in ['-', '/'])
    elif field_name == "alamat":
        symbols = sum(1 for c in text if c in noise_chars and c not in [':', '/', '.', '-'])
    else:
        symbols = sum(1 for c in text if c in noise_chars)
    if len(text) > 0 and (symbols / len(text)) > 0.15:
        return False, "high symbol ratio"
        
    # 5. Alphanumeric Density (excluding spaces and structural symbols for alamat)
    non_space = re.sub(r'\s+', '', text)
    if len(non_space) > 0:
        if field_name == "alamat":
            alnum_count = sum(1 for c in non_space if c.isalnum() or c in [':', '/', '.', '-'])
        else:
            alnum_count = sum(1 for c in non_space if c.isalnum())
        if (alnum_count / len(non_space)) < 0.70:
            return False, "low alphanumeric density"

    # 6. Garbage Short-Word Ratio & OCR Hallucination Detection (for nama, alamat, kelurahan_desa, kecamatan)
    #    Pattern: "WIN HADIAY J" -> 3 kata, 2 kata pendek (≤1 huruf)
    #    Pattern: "SERN TAW SD TAI AN TTR OO DEDEN KUSMANI" -> banyak kata 2-3 huruf random
    #    Pattern: "TR TORE THE WCW WER" -> English noise words from Tesseract background hallucination
    if field_name in ["nama", "kelurahan_desa", "kecamatan", "alamat"]:
        words = text_upper.split()
        
        # English OCR noise words commonly hallucinated by Tesseract on card textures
        english_ocr_noise = {
            "THE", "WCW", "WER", "TORE", "TTR", "WAS", "HAS", "WITH", "FROM",
            "THIS", "THAT", "THEM", "THEIR", "HAVE", "HAD", "BEEN", "WILL", "WOULD"
        }
        for w in words:
            if w in english_ocr_noise:
                return False, f"contains English OCR noise word '{w}'"

        # Check assess_name_quality for nama
        if field_name == "nama":
            from app.ktp.extractor.identity import assess_name_quality
            if not assess_name_quality(text):
                return False, "nama failed assess_name_quality"

        # Check single-char garbage words & short garbage words for nama, kelurahan_desa, kecamatan
        # (Addresses are excluded because they legitimately contain 'NO', 'RT', 'RW', block letters like 'E', '4')
        if field_name in ["nama", "kelurahan_desa", "kecamatan"]:
            single_char_words = sum(1 for w in words if len(w) <= 1)
            if len(words) >= 2 and single_char_words / len(words) >= 0.33:
                return False, f"too many single-char garbage words ({single_char_words}/{len(words)})"
            
            if len(words) >= 3:
                short_garbage = sum(1 for w in words if len(w) <= 2)
                if short_garbage / len(words) > 0.40:
                    return False, f"too many short garbage words ({short_garbage}/{len(words)})"
            
            if len(words) >= 2:
                avg_word_len = sum(len(w) for w in words) / len(words)
                if avg_word_len < 3.0:
                    return False, f"average word length too short ({avg_word_len:.1f})"
        
        # Check for Pekerjaan keywords leaking into Kecamatan
        if field_name == "kecamatan" and any(k in text_upper for k in ["SWASTA", "BURUH", "PNS", "KARYAWAN", "WIRASWASTA", "PETANI", "PELAJAR"]):
            return False, "kecamatan contains pekerjaan keyword"

        # Check for Pekerjaan gibberish (e.g. "gta EAP")
        if field_name == "pekerjaan":
            if any(k in text_upper for k in ["BURUH", "KARYAWAN", "PNS", "SWASTA", "WIRASWASTA", "PETANI", "PELAJAR", "IBU RUMAH TANGGA", "PNS", "TNI", "POLRI"]):
                pass
            elif len(text) < 5 or any(bad in text_upper for bad in ["GTA", "EAP"]):
                return False, "pekerjaan text invalid or gibberish"

        # Check for Nama gibberish or too short (< 3 chars like "AE")
        if field_name == "nama" and len(text) < 3:
            return False, f"nama too short ({len(text)} chars)"
        if field_name == "nama" and len(words) > 6:
            return False, f"nama has too many words ({len(words)})"
            
    return True, ""


def merge_roi_and_fallback_extract(
    roi_results: dict, 
    general_parsed_data: object,
    general_base_score: int,
    general_raw_text: str,
    general_word_conf_map: dict,
    field_specific_raw_text: dict = None,
    field_specific_word_map: dict = None
) -> dict:
    """
    Merge untuk endpoint /v1/extract. 
    Membandingkan ROI vs General OCR murni.
    Prioritas utama diberikan kepada ROI jika ROI membaca nilai yang valid (sane).
    """
    if field_specific_raw_text is None:
        field_specific_raw_text = {}
    if field_specific_word_map is None:
        field_specific_word_map = {}

    merged = {}
    field_mappings = [
        "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
        "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
        "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
    ]
    
    roi_all_fields_temp = {f: roi_results.get(f, {}).get("raw_text", "") for f in field_mappings}
    gen_all_fields_temp = {f: getattr(general_parsed_data, f, "") for f in field_mappings}
    
    for field in field_mappings:
        roi_res = roi_results.get(field, {})
        roi_text = roi_res.get("raw_text", "")
        roi_extracted = roi_res.get("extracted_value", "")
        roi_raw_conf = roi_res.get("confidence", 0.0)
        roi_word_conf = roi_res.get("word_conf_map", {})
        
        general_text = getattr(general_parsed_data, field, None)
        if field == "nik" and general_text:
            from app.ktp.extractor.validators import sync_nik_with_birthdate
            synced = sync_nik_with_birthdate(
                str(general_text),
                getattr(general_parsed_data, "tanggal_lahir", None),
                getattr(general_parsed_data, "jenis_kelamin", None)
            )
            if synced:
                general_text = synced
        
        roi_calibrated_conf = calculate_field_confidence(
            field_name=field,
            value=roi_text,
            base_score=int(roi_raw_conf),
            word_conf_map=roi_word_conf,
            raw_text=roi_text,
            all_fields=roi_all_fields_temp
        )
        
        specific_raw_text = field_specific_raw_text.get(field, general_raw_text)
        specific_word_map = field_specific_word_map.get(field, general_word_conf_map)

        gen_calibrated_conf = calculate_field_confidence(
            field_name=field,
            value=general_text,
            base_score=general_base_score,
            word_conf_map=specific_word_map,
            raw_text=specific_raw_text,
            all_fields=gen_all_fields_temp
        )
        
        final_roi_value = roi_extracted
        if not final_roi_value or len(str(final_roi_value).strip()) < 2:
            if field == "nik":
                match = re.search(r'\b\d{16}\b', roi_text)
                if match:
                    final_roi_value = match.group(0)
            elif field == "tanggal_lahir":
                match = re.search(r'\d{2}-\d{2}-\d{4}', roi_text)
                if match:
                    final_roi_value = match.group(0)
            elif field == "rt_rw":
                match = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', roi_text)
                if match:
                    final_roi_value = f"{int(match.group(1)):03d}/{int(match.group(2)):03d}"
            else:
                final_roi_value = roi_text

        is_roi_valid = False
        if final_roi_value and len(str(final_roi_value).strip()) >= 2:
            is_sane, reason = _sanity_check_free_text(field, str(final_roi_value))
            if is_sane:
                is_roi_valid = True
            else:
                logger.info(f"[ROI DEBUG] Field '{field}' rejected by sanity check: {reason}")

        # Smart Safeguard Lock for all fields in extract merge:
        # If General OCR produced a valid sane value, prefer General OCR unless ROI has strictly higher confidence
        if general_text:
            if field == "nama":
                general_text = re.sub(r'^\b(KAMA|NAMA|NAME)\b[\s:\._\-]*', '', str(general_text), flags=re.IGNORECASE).strip()
            is_gen_sane, _ = _sanity_check_free_text(field, str(general_text))
            if is_gen_sane and (not is_roi_valid or gen_calibrated_conf >= roi_calibrated_conf):
                is_roi_valid = False

        if is_roi_valid:
            conf_out = max(roi_calibrated_conf, 82.0)
            merged[field] = FieldWithSource(
                value=str(final_roi_value).strip(),
                confidence=conf_out,
                source="ROI"
            )
        else:
            merged[field] = FieldWithSource(
                value=general_text if general_text else None,
                confidence=gen_calibrated_conf, 
                source="GENERAL"
            )

    # Post-Merge Validasi NIK dengan Tanggal Lahir & Jenis Kelamin
    from app.ktp.extractor import validators
    merged_nik = merged.get("nik").value if merged.get("nik") else None
    merged_tgl = merged.get("tanggal_lahir").value if merged.get("tanggal_lahir") else None
    merged_jk = merged.get("jenis_kelamin").value if merged.get("jenis_kelamin") else None

    if merged_nik and len(merged_nik) == 16 and merged_nik.isdigit():
        try:
            dd = int(merged_nik[6:8])
            is_perempuan = dd > 40
            if is_perempuan:
                dd -= 40
            mm = int(merged_nik[8:10])
            yy_short = int(merged_nik[10:12])
            yy_full = (1900 + yy_short) if yy_short > 26 else (2000 + yy_short)

            import datetime
            if 1 <= dd <= 31 and 1 <= mm <= 12:
                datetime.date(yy_full, mm, dd)
                inferred_dob = f"{dd:02d}-{mm:02d}-{yy_full:04d}"
                if not merged_tgl or merged_tgl != inferred_dob:
                    merged["tanggal_lahir"] = FieldWithSource(value=inferred_dob, confidence=90.0, source="GENERAL")
                    merged_tgl = inferred_dob

                if not merged.get("jenis_kelamin") or not merged["jenis_kelamin"].value:
                    inferred_jk = "PEREMPUAN" if is_perempuan else "LAKI-LAKI"
                    merged["jenis_kelamin"] = FieldWithSource(value=inferred_jk, confidence=88.0, source="GENERAL")

                if not merged.get("kewarganegaraan") or not merged["kewarganegaraan"].value:
                    merged["kewarganegaraan"] = FieldWithSource(value="WNI", confidence=85.0, source="GENERAL")
        except ValueError:
            pass

    if merged_nik and merged_tgl:
        synced_nik = validators.sync_nik_with_birthdate(merged_nik, merged_tgl, merged_jk)
        if synced_nik and synced_nik != merged_nik:
            merged["nik"] = FieldWithSource(value=synced_nik, confidence=99.0, source="GENERAL")

    if not merged.get("golongan_darah") or not merged["golongan_darah"].value:
        merged["golongan_darah"] = FieldWithSource(value="-", confidence=88.0, source="GENERAL")

    return merged


def merge_roi_and_fallback_validate(
    roi_results: dict,
    consensus_general_data: dict
) -> dict:
    """
    Merge untuk endpoint /v1/validate.
    Prioritas utama diberikan kepada ROI jika ROI membaca nilai yang valid (sane).
    """
    merged = {}
    field_mappings = [
        "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
        "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
        "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
    ]
    
    roi_all_fields_temp = {f: roi_results.get(f, {}).get("raw_text", "") for f in field_mappings}
    
    for field in field_mappings:
        roi_res = roi_results.get(field, {})
        roi_text = roi_res.get("raw_text", "")
        roi_extracted = roi_res.get("extracted_value", "")
        roi_raw_conf = roi_res.get("confidence", 0.0)
        roi_word_conf = roi_res.get("word_conf_map", {})
        
        cg = consensus_general_data.get(field, {})
        cg_val = cg.get("value")
        if field == "alamat" and cg_val:
            from app.ktp.extractor.address import extract_alamat
            cleaned_addr = extract_alamat(cg_val)
            if cleaned_addr:
                cg_val = cleaned_addr

        cg_conf = cg.get("confidence", 0.0)
        cg_source = cg.get("source", "GENERAL")
        
        mapped_source = "CONSENSUS" if "mobile" in cg_source else "GENERAL"
        
        roi_calibrated_conf = calculate_field_confidence(
            field_name=field,
            value=roi_text,
            base_score=int(roi_raw_conf),
            word_conf_map=roi_word_conf,
            raw_text=roi_text,
            all_fields=roi_all_fields_temp
        )
        
        final_roi_value = roi_extracted
        if not final_roi_value:
            if field == "nik":
                cleaned = re.sub(r'\D', '', roi_text)
                if len(cleaned) == 16:
                    final_roi_value = cleaned
            elif field == "tanggal_lahir":
                match = re.search(r'\d{2}-\d{2}-\d{4}', roi_text)
                if match:
                    final_roi_value = match.group(0)
            elif field == "rt_rw":
                match = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', roi_text)
                if match:
                    final_roi_value = f"{int(match.group(1)):03d}/{int(match.group(2)):03d}"
            else:
                final_roi_value = roi_text

        is_roi_valid = False
        if final_roi_value and len(str(final_roi_value).strip()) >= 2:
            is_sane, reason = _sanity_check_free_text(field, str(final_roi_value))
            if is_sane:
                is_roi_valid = True

        # ROI Garble Protection ketika cg_val null:
        # Jika tidak ada nilai dari consensus/mobile (cg_val None), naikkan threshold ROI ke 90.0
        # dan tolak ROI yang mengandung terlalu banyak noise token (>30% non-alpha)
        if is_roi_valid and not cg_val:
            _STRICT_ROI_FIELDS = {"nama", "tempat_lahir", "alamat", "kelurahan_desa", "kecamatan", "agama", "status_perkawinan", "pekerjaan"}
            if field in _STRICT_ROI_FIELDS:
                # Naikkan threshold
                if roi_calibrated_conf < 90.0:
                    is_roi_valid = False
                else:
                    # Cek noise token ratio
                    roi_tokens = str(final_roi_value).split()
                    noise_count = sum(1 for t in roi_tokens if not re.search(r'[A-Za-z0-9]', t))
                    if roi_tokens and (noise_count / len(roi_tokens)) > 0.30:
                        is_roi_valid = False

        # Safeguard Lock NIK: Dynamic verification with DOB confusion sync
        if field == "nik" and cg_val:
            from app.ktp.extractor.validators import validate_nik_structure, sync_nik_with_birthdate
            cleaned_cg_nik = re.sub(r'\D', '', str(cg_val))
            if len(cleaned_cg_nik) == 16:
                cg_val = cleaned_cg_nik
                cg_dob = consensus_general_data.get("tanggal_lahir", {}).get("value") if isinstance(consensus_general_data, dict) else None
                cg_jk = consensus_general_data.get("jenis_kelamin", {}).get("value") if isinstance(consensus_general_data, dict) else None
                if cg_dob:
                    synced_nik = sync_nik_with_birthdate(cg_val, cg_dob, cg_jk)
                    if synced_nik and validate_nik_structure(synced_nik):
                        cg_val = synced_nik
                if validate_nik_structure(cg_val):
                    is_roi_valid = False

        # Smart Safeguard Lock for all other fields in validate merge:
        # If consensus/mobile_data produced a valid sane value, protect it unless ROI has strictly higher confidence
        if field != "nik" and cg_val:
            if field == "golongan_darah" and str(cg_val).strip() == "-" and final_roi_value in ["A", "B", "AB", "O"]:
                pass  # Allow valid blood type to override default '-'
            else:
                if field == "nama" and cg_val:
                    cg_val = re.sub(r'^\b(KAMA|NAMA|NAME)\b[\s:\._\-]*', '', str(cg_val), flags=re.IGNORECASE).strip()
                is_cg_sane, _ = _sanity_check_free_text(field, str(cg_val))
                if is_cg_sane:
                    is_roi_valid = False

        if is_roi_valid:
            conf_out = max(roi_calibrated_conf, 82.0)
            merged[field] = FieldWithSource(
                value=str(final_roi_value).strip(),
                confidence=conf_out,
                source="ROI"
            )
        else:
            merged[field] = FieldWithSource(
                value=cg_val if cg_val else None,
                confidence=cg_conf,
                source=mapped_source
            )

    # Post-processing: Length Sanity Check & Agama Whitelist
    _AGAMA_WHITELIST = {"ISLAM", "KRISTEN", "KATHOLIK", "KATOLIK", "HINDU", "BUDHA", "BUDDHA", "KONGHUCU"}
    _NONSENSICAL_FIELDS = {"nama", "tempat_lahir", "kelurahan_desa", "kecamatan"}
    for _field, _fobj in merged.items():
        if _fobj and _fobj.value:
            _val = str(_fobj.value).strip()
            # Field pendek (< 3 karakter) pada field terstruktur → null
            if _field in _NONSENSICAL_FIELDS and len(_val) < 3:
                merged[_field] = FieldWithSource(value=None, confidence=0.0, source="GENERAL")
                continue
            # kecamatan/kelurahan: >50% token non-alpha → null
            if _field in {"kecamatan", "kelurahan_desa"}:
                _tokens = _val.split()
                _noise = sum(1 for t in _tokens if not re.sub(r'[^A-Za-z]', '', t))
                if _tokens and (_noise / len(_tokens)) > 0.5:
                    merged[_field] = FieldWithSource(value=None, confidence=0.0, source="GENERAL")
                    continue
            # Agama whitelist validation
            if _field == "agama":
                _normalized = re.sub(r'[^A-Z]', '', _val.upper())
                _match = any(_normalized in a.replace(" ", "") or a.replace(" ", "") in _normalized for a in _AGAMA_WHITELIST)
                if not _match:
                    merged[_field] = FieldWithSource(value=None, confidence=0.0, source="GENERAL")
                
    # Post-Merge Validasi NIK dengan Tanggal Lahir & Jenis Kelamin pada Validate
    from app.ktp.extractor import validators
    merged_nik = merged.get("nik").value if merged.get("nik") else None
    merged_tgl = merged.get("tanggal_lahir").value if merged.get("tanggal_lahir") else None
    merged_jk = merged.get("jenis_kelamin").value if merged.get("jenis_kelamin") else None

    if merged_nik and len(merged_nik) == 16 and merged_nik.isdigit():
        try:
            dd = int(merged_nik[6:8])
            is_perempuan = dd > 40
            if is_perempuan:
                dd -= 40
            mm = int(merged_nik[8:10])
            yy_short = int(merged_nik[10:12])
            yy_full = (1900 + yy_short) if yy_short > 26 else (2000 + yy_short)

            import datetime
            if 1 <= dd <= 31 and 1 <= mm <= 12:
                datetime.date(yy_full, mm, dd)
                inferred_dob = f"{dd:02d}-{mm:02d}-{yy_full:04d}"
                if not merged_tgl:
                    merged["tanggal_lahir"] = FieldWithSource(value=inferred_dob, confidence=90.0, source="CONSENSUS")
                    merged_tgl = inferred_dob

                if not merged.get("jenis_kelamin") or not merged["jenis_kelamin"].value:
                    inferred_jk = "PEREMPUAN" if is_perempuan else "LAKI-LAKI"
                    merged["jenis_kelamin"] = FieldWithSource(value=inferred_jk, confidence=88.0, source="CONSENSUS")

                if not merged.get("kewarganegaraan") or not merged["kewarganegaraan"].value:
                    merged["kewarganegaraan"] = FieldWithSource(value="WNI", confidence=85.0, source="CONSENSUS")
        except ValueError:
            pass

    # Clean up berlaku_hingga if contaminated by DOB
    if merged.get("berlaku_hingga") and merged.get("berlaku_hingga").value == merged_tgl:
        merged["berlaku_hingga"] = FieldWithSource(value="SEUMUR HIDUP", confidence=88.0, source="CONSENSUS")

    # Final Fallback Safeguards:
    # 1. berlaku_hingga: default to SEUMUR HIDUP if null
    if not merged.get("berlaku_hingga") or not merged["berlaku_hingga"].value:
        merged["berlaku_hingga"] = FieldWithSource(value="SEUMUR HIDUP", confidence=85.0, source="CONSENSUS")

    # 2. golongan_darah: use ROI/consensus blood type if present, else default '-'
    if not merged.get("golongan_darah") or not merged["golongan_darah"].value or merged["golongan_darah"].value in ["-", "NONE", "NULL", "None"]:
        roi_gol = roi_results.get("golongan_darah", {}).get("raw_text") if roi_results else None
        cg_gol = consensus_general_data.get("golongan_darah", {}).get("value") if isinstance(consensus_general_data, dict) else None
        if roi_gol and roi_gol.strip().upper() in ["A", "B", "AB", "O"]:
            merged["golongan_darah"] = FieldWithSource(value=roi_gol.strip().upper(), confidence=86.8, source="ROI")
        elif cg_gol and cg_gol.strip().upper() in ["A", "B", "AB", "O"]:
            merged["golongan_darah"] = FieldWithSource(value=cg_gol.strip().upper(), confidence=86.8, source="CONSENSUS")
        else:
            merged["golongan_darah"] = FieldWithSource(value="-", confidence=88.0, source="GENERAL")

    # 3. Generic NIK-DOB Synchronization

    merged_nik = merged.get("nik").value if merged.get("nik") else None
    merged_tgl = merged.get("tanggal_lahir").value if merged.get("tanggal_lahir") else None
    merged_jk = merged.get("jenis_kelamin").value if merged.get("jenis_kelamin") else None
    if merged_nik and merged_tgl:
        synced_nik = validators.sync_nik_with_birthdate(merged_nik, merged_tgl, merged_jk)
        if synced_nik and synced_nik != merged_nik:
            old_src = merged["nik"].source
            merged["nik"] = FieldWithSource(value=synced_nik, confidence=99.0, source=old_src)
    # 4. Regional Hierarchy Fallback for kecamatan if null or garbled
    merged_kec = merged.get("kecamatan").value if merged.get("kecamatan") else None
    merged_kel = merged.get("kelurahan_desa").value if merged.get("kelurahan_desa") else None
    if (not merged_kec or len(merged_kec.strip()) < 3) and merged_kel:
        from app.ktp.v1.regional_normalizer import lookup_regional_hierarchy
        inferred = lookup_regional_hierarchy(kelurahan_desa=merged_kel)
        if inferred and inferred.get("kecamatan"):
            merged["kecamatan"] = FieldWithSource(value=inferred["kecamatan"], confidence=88.0, source="CONSENSUS")

    return merged
