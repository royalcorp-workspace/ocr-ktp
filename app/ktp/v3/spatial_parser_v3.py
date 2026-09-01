import re
from typing import List, Dict, Any, Tuple, Optional, Set
from app.ktp.v2.paddle_engine import PaddleTextBox
from app.ktp.v3.field_cleaners_v3 import (
    clean_text, clean_nik, clean_date, clean_gender,
    clean_blood_type, clean_marital_status, clean_citizenship,
    clean_rt_rw, tokenize_pekerjaan, normalize_regional, tokenize_compound_name,
    tokenize_name, tokenize_address, extract_date_from_nik, _V3_NAME_LEXICON,
    _KOTA_KABUPATEN_LEXICON
)

# Label Anchor Matrix Patterns with OCR Typo Tolerance for V3
LABEL_PATTERNS = {
    "nik": [r'\b(NIK|HIK|MIL|NlK|N1K)\b'],
    "nama": [r'\b(NAMA|NAM)\b'],
    "tempat_tanggal_lahir": [
        r'TEMPA[TL]\s*/?\s*TG[IL1]\s*LAHIR',
        r'TEMPA[TL]\s+LAHIR',
        r'TG[IL1]\s*LAHIR',
        r'\bLAHIR\b',
        r'\bTEMPAT\b',
        r'\bat/Tgl\s*Lahir\b',
        r'\bTgl\s*Lahir\b'
    ],
    "jenis_kelamin": [
        r'J[EO]N[I1]S\s+KE[IL1]AM[I1][NM]',
        r'JENIS\s+KELAMI[NM]',
        r'\bKE[IL1]AM[I1][NM]\b',
        r'\bJ[EO]N[I1]S\b',
        r'\bxelamin\b',
        r'\bkelamin\b'
    ],
    "golongan_darah": [r'GOL\.?\s*DARAH', r'\bDARAH\b', r'\bGOL\b'],
    "alamat": [r'\b(ALAMAT|ALAMA|ALMT)\b'],
    "rt_rw": [r'\b(RT\s*/?\s*RW|RT|RW|RI/RW|RT/RV|RT/PW|T/RW)\b'],
    "kelurahan_desa": [
        r'KE[LV1I/|]\s*[/.\s]?\s*DESA',
        r'\bKE[LV]DESA\b',
        r'KEL\s*/?\s*DESA',
        r'\bDESA\b',
        r'\bKELURAHAN\b',
        r'\bel/Desa\b'
    ],
    "kecamatan": [
        r'\b(KECAMATAN|KECAM|KEC)\b',
        r'\b[KkE]?CAMATAN\b',
        r'\bECAMATAN\b'
    ],
    "agama": [r'\bAGAMA\b'],
    "status_perkawinan": [
        r'STATUS\s+PERKAWINA[NR]',
        r'STATUS\s+PERKAWINAN',
        r'\bPERKAWINA[NR]\b',
        r'\bPerkawinan\b'
    ],
    "pekerjaan": [r'\b(PEKERJAAN|PEKERJA)\b'],
    "kewarganegaraan": [r'\b(KEWARGANEGARAAN|KEWARGAN|anegaraan)\b'],
    "berlaku_hingga": [r'.*RLAKU\s*HINGGA', r'.*HINGGA', r'\b[BN]ERLAKU\b', r'\bHingga\b'],
}


def is_label_text(text: str) -> Optional[str]:
    """Returns label_key if text matches any label pattern, else None."""
    upper = text.upper().strip()
    for label_key, patterns in LABEL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, upper):
                return label_key
    return None


def extract_inline_value(text: str, label_key: str) -> Optional[str]:
    """Extracts value text if label and value are combined in a single box."""
    if not text:
        return None
    s = text.strip()

    if ":" in s:
        parts = s.split(":", 1)
        val = parts[1].strip()
        if val:
            return val

    patterns = LABEL_PATTERNS.get(label_key, [])
    upper = s.upper()
    for pat in patterns:
        m = re.search(pat, upper)
        if m:
            val = s[m.end():].strip()
            if val:
                return val
    return None


def group_boxes_into_lines(boxes: List[PaddleTextBox]) -> List[List[PaddleTextBox]]:
    """Groups text boxes into horizontal lines based on vertical overlap."""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.center_y)
    lines: List[List[PaddleTextBox]] = []

    for box in sorted_boxes:
        placed = False
        for line in lines:
            avg_y = sum(b.center_y for b in line) / len(line)
            min_h = min(b.height for b in line)
            threshold = max(8.0, 0.5 * min_h)

            if abs(box.center_y - avg_y) < threshold:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda b: b.x_min)

    lines = sorted(lines, key=lambda line: sum(b.center_y for b in line) / len(line))
    return lines


class SpatialParserV3:
    def parse_ktp(self, text_boxes: List[PaddleTextBox]) -> Dict[str, Dict[str, Any]]:
        """
        Parses text boxes using 5-Layer Multi-Strategy Pipeline with Deterministic Spatial Slot Grid.
        """
        if not text_boxes:
            return {k: {"val": None, "conf": 0.0} for k in [
                "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
                "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
                "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
            ]}

        w_max = max(b.x_max for b in text_boxes)
        h_max = max(b.y_max for b in text_boxes)

        lines = group_boxes_into_lines(text_boxes)

        extracted_raw: Dict[str, Dict[str, Any]] = {
            "nik": {"val": None, "conf": 0.0},
            "nama": {"val": None, "conf": 0.0},
            "tempat_lahir": {"val": None, "conf": 0.0},
            "tanggal_lahir": {"val": None, "conf": 0.0},
            "jenis_kelamin": {"val": None, "conf": 0.0},
            "golongan_darah": {"val": None, "conf": 0.0},
            "alamat": {"val": None, "conf": 0.0},
            "rt_rw": {"val": None, "conf": 0.0},
            "kelurahan_desa": {"val": None, "conf": 0.0},
            "kecamatan": {"val": None, "conf": 0.0},
            "agama": {"val": None, "conf": 0.0},
            "status_perkawinan": {"val": None, "conf": 0.0},
            "pekerjaan": {"val": None, "conf": 0.0},
            "kewarganegaraan": {"val": None, "conf": 0.0},
            "berlaku_hingga": {"val": None, "conf": 0.0},
        }

        # Track exact used bounding box IDs to guarantee zero cross-field duplication
        used_box_ids: Set[int] = set()
        # Track vertical anchors of key landmark fields
        field_y_anchors: Dict[str, float] = {}

        def is_signature_zone(b: PaddleTextBox) -> bool:
            return b.x_min > (0.55 * w_max) and b.y_min > (0.55 * h_max)

        for line in lines:
            line_str = " ".join(b.text for b in line)
            line_avg_y = sum(b.center_y for b in line) / len(line)
            if line_avg_y < (0.18 * h_max) and ("PROVINSI" in line_str.upper() or "KABUPATEN" in line_str.upper()):
                for b in line:
                    used_box_ids.add(id(b))
                continue

            # 1. Compound Line: TEMPAT/TGL LAHIR
            if is_label_text(line_str) == "tempat_tanggal_lahir" or re.search(r'TEMPA[TL].*LAHIR|TG[IL1].*LAHIR', line_str, re.I):
                val_boxes = [b for b in line if not is_label_text(b.text) and not is_signature_zone(b)]
                val_str = ""
                avg_conf = 0.0

                if val_boxes:
                    val_str = " ".join(b.text for b in val_boxes)
                    avg_conf = sum(b.confidence for b in val_boxes) / len(val_boxes)
                    for b in val_boxes:
                        used_box_ids.add(id(b))
                else:
                    for b in line:
                        inline_v = extract_inline_value(b.text, "tempat_tanggal_lahir")
                        if inline_v:
                            val_str = inline_v
                            avg_conf = b.confidence
                            used_box_ids.add(id(b))
                            break

                if val_str:
                    parts = [p.strip() for p in re.split(r'[,:\.]+', val_str) if p.strip()]
                    if len(parts) >= 2:
                        city_part = re.sub(r'[^A-Z\s]', '', parts[0].upper())
                        date_part = " ".join(parts[1:])
                    elif len(parts) == 1:
                        city_part = re.sub(r'[^A-Z\s]', '', parts[0].upper())
                        date_part = parts[0]
                    else:
                        city_part = ""
                        date_part = ""

                    c_city = normalize_regional(clean_text(city_part))
                    c_date = clean_date(date_part)

                    # Spatial scan on line if date was in a separate right-side box
                    if not c_date:
                        for b in line:
                            if not is_signature_zone(b) and id(b) not in used_box_ids:
                                cd = clean_date(b.text)
                                if cd:
                                    c_date = cd
                                    used_box_ids.add(id(b))
                                    break

                    if c_city:
                        extracted_raw["tempat_lahir"] = {"val": c_city, "conf": round(avg_conf, 1)}
                    if c_date:
                        extracted_raw["tanggal_lahir"] = {"val": c_date, "conf": round(avg_conf, 1)}
                continue

            # 2. Compound Line: JENIS KELAMIN & GOL. DARAH
            if is_label_text(line_str) == "jenis_kelamin" or re.search(r'J[EO]N[I1]S\s+KE[IL1]AM[I1][NM]|JENIS\s+KELAMI[NM]', line_str, re.I):
                gender_boxes = []
                blood_boxes = []
                found_blood_label = False

                for b in line:
                    if is_signature_zone(b):
                        continue
                    if is_label_text(b.text) == "golongan_darah" or "GOL" in b.text.upper():
                        found_blood_label = True
                        used_box_ids.add(id(b))
                        inline_b = extract_inline_value(b.text, "golongan_darah")
                        if inline_b:
                            c_b = clean_blood_type(inline_b)
                            if c_b:
                                extracted_raw["golongan_darah"] = {"val": c_b, "conf": round(b.confidence, 1)}
                        continue
                    if is_label_text(b.text) == "jenis_kelamin":
                        used_box_ids.add(id(b))
                        inline_g = extract_inline_value(b.text, "jenis_kelamin")
                        if inline_g:
                            c_g = clean_gender(inline_g)
                            if c_g:
                                extracted_raw["jenis_kelamin"] = {"val": c_g, "conf": round(b.confidence, 1)}
                        continue

                    if found_blood_label:
                        blood_boxes.append(b)
                    else:
                        gender_boxes.append(b)

                if gender_boxes and not extracted_raw["jenis_kelamin"]["val"]:
                    g_str = " ".join(b.text for b in gender_boxes)
                    g_conf = sum(b.confidence for b in gender_boxes) / len(gender_boxes)
                    c_g = clean_gender(g_str)
                    if c_g:
                        extracted_raw["jenis_kelamin"] = {"val": c_g, "conf": round(g_conf, 1)}
                    for b in gender_boxes:
                        used_box_ids.add(id(b))

                if blood_boxes and not extracted_raw["golongan_darah"]["val"]:
                    b_str = " ".join(b.text for b in blood_boxes)
                    b_conf = sum(b.confidence for b in blood_boxes) / len(blood_boxes)
                    c_b = clean_blood_type(b_str)
                    if c_b:
                        extracted_raw["golongan_darah"] = {"val": c_b, "conf": round(b_conf, 1)}
                    for b in blood_boxes:
                        used_box_ids.add(id(b))
                continue

            # 3. Standard Single-Field Labels
            for idx, box in enumerate(line):
                label_key = is_label_text(box.text)
                if not label_key or label_key in ["tempat_tanggal_lahir", "jenis_kelamin", "golongan_darah"]:
                    continue

                used_box_ids.add(id(box))
                inline_val = extract_inline_value(box.text, label_key)
                val_str = ""
                avg_conf = 0.0

                if inline_val:
                    val_str = inline_val
                    avg_conf = box.confidence
                else:
                    val_boxes = []
                    for right_box in line[idx + 1:]:
                        if is_label_text(right_box.text):
                            break
                        if is_signature_zone(right_box):
                            continue
                        val_boxes.append(right_box)

                    if val_boxes:
                        val_str = " ".join(b.text for b in val_boxes)
                        avg_conf = sum(b.confidence for b in val_boxes) / len(val_boxes)
                        for b in val_boxes:
                            used_box_ids.add(id(b))

                if val_str:
                    if label_key == "nik":
                        c_val = clean_nik(val_str)
                    elif label_key == "nama":
                        c_val = tokenize_name(val_str)
                    elif label_key == "rt_rw":
                        c_val = clean_rt_rw(val_str)
                        field_y_anchors["rt_rw"] = box.center_y
                    elif label_key == "agama":
                        c_val = clean_text(val_str)
                        field_y_anchors["agama"] = box.center_y
                    elif label_key == "status_perkawinan":
                        c_val = clean_marital_status(val_str)
                    elif label_key == "kewarganegaraan":
                        c_val = clean_citizenship(val_str)
                    elif label_key in ("kecamatan", "tempat_lahir"):
                        c_val = normalize_regional(clean_text(val_str))
                        if label_key == "kecamatan":
                            field_y_anchors["kecamatan"] = box.center_y
                    elif label_key == "kelurahan_desa":
                        c_val = normalize_regional(tokenize_compound_name(clean_text(val_str)))
                        field_y_anchors["kelurahan_desa"] = box.center_y
                    elif label_key == "pekerjaan":
                        c_val = tokenize_pekerjaan(clean_text(val_str))
                    elif label_key == "berlaku_hingga":
                        raw_clean = clean_text(val_str)
                        if raw_clean in {"HINGGA", "BERLAKU HINGGA", "BERLAKU", "NORLAKU HINGGA"}:
                            raw_clean = None
                            for right_b in line[idx + 1:]:
                                if re.search(r'SEU[MN]UR', right_b.text.upper()):
                                    raw_clean = "SEUMUR HIDUP"
                                    used_box_ids.add(id(right_b))
                                    break
                                cd = clean_date(right_b.text)
                                if cd:
                                    raw_clean = cd
                                    used_box_ids.add(id(right_b))
                                    break
                        if raw_clean:
                            if re.search(r'SEU[MN]UR', raw_clean):
                                raw_clean = "SEUMUR HIDUP"
                            else:
                                date_m = re.search(r'(\d{2}[-./]\d{2}[-./]\d{4})', raw_clean)
                                if date_m:
                                    raw_clean = clean_date(date_m.group(1))
                        c_val = raw_clean
                    elif label_key == "alamat":
                        line_idx = lines.index(line)
                        extra_parts = [val_str]
                        extra_conf_sum = avg_conf
                        extra_count = 1
                        field_y_anchors["alamat"] = box.center_y
                        for next_line in lines[line_idx + 1:]:
                            if any(is_label_text(b.text) for b in next_line):
                                break
                            next_avg_y = sum(b.center_y for b in next_line) / len(next_line)
                            cur_avg_y = sum(b.center_y for b in line) / len(line)
                            if next_avg_y - cur_avg_y > 80:
                                break
                            non_sig_boxes = [b for b in next_line if not is_signature_zone(b)]
                            if non_sig_boxes:
                                extra_parts.append(" ".join(b.text for b in non_sig_boxes))
                                extra_conf_sum += sum(b.confidence for b in non_sig_boxes) / len(non_sig_boxes)
                                extra_count += 1
                                for b in non_sig_boxes:
                                    used_box_ids.add(id(b))
                        full_alamat = " ".join(extra_parts)
                        # Extract inline RT/RW if combined with address
                        rtrw_inline = re.search(r'(R[TI][/\s]*RW\s*:?\s*\d+/\d+|\b\d{2,3}/\d{2,3}\b)', full_alamat, re.I)
                        if rtrw_inline and not extracted_raw["rt_rw"]["val"]:
                            raw_inline_rtrw = rtrw_inline.group(0)
                            c_rtrw = clean_rt_rw(raw_inline_rtrw)
                            if c_rtrw:
                                extracted_raw["rt_rw"] = {"val": c_rtrw, "conf": round(avg_conf, 1)}
                                full_alamat = full_alamat.replace(raw_inline_rtrw, "").strip()
                        c_val = tokenize_address(full_alamat)
                        avg_conf = extra_conf_sum / extra_count
                    else:
                        c_val = clean_text(val_str)

                    if c_val:
                        extracted_raw[label_key] = {"val": c_val, "conf": round(avg_conf, 1)}

        # Fallback 1: NIK Rescue across all text boxes
        if not extracted_raw["nik"]["val"]:
            for b in text_boxes:
                possible_nik = clean_nik(b.text)
                if possible_nik and len(possible_nik) == 16:
                    extracted_raw["nik"] = {"val": possible_nik, "conf": round(b.confidence, 1)}
                    used_box_ids.add(id(b))
                    break

        # Fallback 2: Positional Structural Fallback
        if not extracted_raw["nama"]["val"] or not extracted_raw["jenis_kelamin"]["val"]:
            body_boxes = [b for b in text_boxes if b.y_min > (0.18 * h_max) and not is_signature_zone(b) and id(b) not in used_box_ids]
            body_boxes = sorted(body_boxes, key=lambda b: b.y_min)

            for b in body_boxes:
                txt = b.text.strip()
                if not txt:
                    continue

                if not extracted_raw["jenis_kelamin"]["val"]:
                    cg = clean_gender(txt)
                    if cg:
                        extracted_raw["jenis_kelamin"] = {"val": cg, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                        continue

                if not extracted_raw["agama"]["val"]:
                    clean_ag = clean_text(txt)
                    if clean_ag and clean_ag.upper() in ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KHONGHUCU"]:
                        extracted_raw["agama"] = {"val": clean_ag.upper(), "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                        field_y_anchors["agama"] = b.center_y
                        continue

                if not extracted_raw["status_perkawinan"]["val"]:
                    cs = clean_marital_status(txt)
                    if cs:
                        extracted_raw["status_perkawinan"] = {"val": cs, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                        continue

                if not extracted_raw["kewarganegaraan"]["val"]:
                    ck = clean_citizenship(txt)
                    if ck:
                        extracted_raw["kewarganegaraan"] = {"val": ck, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                        continue

                if not extracted_raw["pekerjaan"]["val"]:
                    if "BURUH" in txt.upper() or "PELAJAR" in txt.upper() or "SWASTA" in txt.upper() or "PNS" in txt.upper():
                        extracted_raw["pekerjaan"] = {"val": tokenize_pekerjaan(txt), "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                        continue

                if not extracted_raw["berlaku_hingga"]["val"]:
                    cd = clean_date(txt)
                    is_city_date = bool(re.search(r'[A-Z]{3,}[^\d]*\d{2}[\-\./]\d{2}[\-\./]\d{4}', txt))
                    if "SEUMUR" in txt.upper():
                        extracted_raw["berlaku_hingga"] = {"val": "SEUMUR HIDUP", "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                        continue
                    elif cd and not is_city_date:
                        extracted_raw["berlaku_hingga"] = {"val": cd, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                        continue

                if (not extracted_raw["tempat_lahir"]["val"] or not extracted_raw["tanggal_lahir"]["val"]) and clean_date(txt):
                    parts = re.split(r'[,:\.]+', txt)
                    if len(parts) >= 2:
                        c_city = normalize_regional(clean_text(parts[0]))
                        c_date = clean_date(" ".join(parts[1:]))
                        if c_city:
                            extracted_raw["tempat_lahir"] = {"val": c_city, "conf": round(b.confidence, 1)}
                        if c_date:
                            extracted_raw["tanggal_lahir"] = {"val": c_date, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                    continue

                if not extracted_raw["nama"]["val"] and not is_label_text(txt) and len(txt) > 3 and not re.search(r'\d', txt):
                    extracted_raw["nama"] = {"val": tokenize_name(txt), "conf": round(b.confidence, 1)}
                    used_box_ids.add(id(b))

        # Layer 5: Pattern-Based Field Guesser with Deterministic Spatial Slot Grid
        self._layer5_pattern_guesser(text_boxes, extracted_raw, h_max, w_max, used_box_ids, field_y_anchors)

        # Fallback 3: Dukcapil Date of Birth Recovery from NIK
        if not extracted_raw["tanggal_lahir"]["val"] and extracted_raw["nik"]["val"]:
            nik_val = extracted_raw["nik"]["val"]
            nik_date = extract_date_from_nik(nik_val)
            if nik_date:
                nik_conf = extracted_raw["nik"].get("conf", 95.0)
                extracted_raw["tanggal_lahir"] = {"val": nik_date, "conf": round(nik_conf, 1)}

        # Fallback 4: Indonesian e-KTP Lifetime Expiry Rule (UU No. 24 Tahun 2013)
        if not extracted_raw["berlaku_hingga"]["val"] and extracted_raw["nik"]["val"]:
            nik_conf = extracted_raw["nik"].get("conf", 95.0)
            extracted_raw["berlaku_hingga"] = {"val": "SEUMUR HIDUP", "conf": round(nik_conf, 1)}

        return extracted_raw

    def _layer5_pattern_guesser(
        self,
        text_boxes: List[PaddleTextBox],
        extracted_raw: Dict[str, Dict[str, Any]],
        h_max: float,
        w_max: float,
        used_box_ids: Set[int],
        field_y_anchors: Dict[str, float]
    ) -> None:
        def is_signature_zone(b: PaddleTextBox) -> bool:
            return b.x_min > (0.55 * w_max) and b.y_min > (0.55 * h_max)

        assigned_vals = {item["val"] for item in extracted_raw.values() if item and item.get("val")}
        body_boxes = [b for b in text_boxes if b.y_min > (0.18 * h_max) and not is_signature_zone(b)]

        unassigned: List[PaddleTextBox] = []
        for b in body_boxes:
            txt_clean = clean_text(b.text)
            if not txt_clean:
                continue
            # Strictly exclude boxes that have already been assigned to another field
            if id(b) in used_box_ids or txt_clean in assigned_vals or is_label_text(b.text):
                continue
            unassigned.append(b)

        unassigned = sorted(unassigned, key=lambda b: b.y_min)
        kelurahan_candidates: List[PaddleTextBox] = []

        y_rtrw = field_y_anchors.get("rt_rw")
        y_agama = field_y_anchors.get("agama")

        for b in unassigned:
            txt = clean_text(b.text) or ""

            # 1. Jenis kelamin
            if not extracted_raw["jenis_kelamin"]["val"]:
                cg = clean_gender(txt)
                if cg:
                    extracted_raw["jenis_kelamin"] = {"val": cg, "conf": round(b.confidence, 1)}
                    assigned_vals.add(cg)
                    used_box_ids.add(id(b))
                    continue

            # 2. RT/RW
            if not extracted_raw["rt_rw"]["val"]:
                clean_rtrw = clean_rt_rw(txt)
                if clean_rtrw and re.match(r'^\d{1,3}/\d{1,3}$', clean_rtrw):
                    extracted_raw["rt_rw"] = {"val": clean_rtrw, "conf": round(b.confidence, 1)}
                    assigned_vals.add(clean_rtrw)
                    used_box_ids.add(id(b))
                    y_rtrw = b.center_y
                    continue

            # 3. Alamat
            if not extracted_raw["alamat"]["val"]:
                if re.match(r'^(KP\.|JL\.|KMP\.|PERUM\.|GG\.|DSN\.|BLOK|KP\s|JL\s)', txt, re.I):
                    extracted_raw["alamat"] = {"val": txt, "conf": round(b.confidence, 1)}
                    assigned_vals.add(txt)
                    used_box_ids.add(id(b))
                    continue

            # 4. Agama
            if not extracted_raw["agama"]["val"]:
                if txt.upper() in ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KHONGHUCU"]:
                    extracted_raw["agama"] = {"val": txt.upper(), "conf": round(b.confidence, 1)}
                    assigned_vals.add(txt.upper())
                    used_box_ids.add(id(b))
                    y_agama = b.center_y
                    continue

            # 5. Berlaku hingga scan in bottom zone
            if not extracted_raw["berlaku_hingga"]["val"] and b.y_min > (0.55 * h_max):
                if "SEUMUR" in txt.upper() or "HIDUP" in txt.upper():
                    extracted_raw["berlaku_hingga"] = {"val": "SEUMUR HIDUP", "conf": round(b.confidence, 1)}
                    used_box_ids.add(id(b))
                    continue
                cd = clean_date(txt)
                is_city_date = bool(re.search(r'[A-Z]{3,}[^\d]*\d{2}[\-\./]\d{2}[\-\./]\d{4}', txt))
                if cd and not is_city_date:
                    extracted_raw["berlaku_hingga"] = {"val": cd, "conf": round(b.confidence, 1)}
                    used_box_ids.add(id(b))
                    continue

            # 6. Kelurahan/Kecamatan Candidates with Template Y-Band Spatial Constraints
            if not extracted_raw["kelurahan_desa"]["val"] or not extracted_raw["kecamatan"]["val"]:
                if re.match(r'^[A-Z\s\.\-]+$', txt) and len(txt) > 3 and not re.search(r'\d', txt):
                    noise_terms = {
                        "PROVINSI", "KABUPATEN", "LAKI-LAKI", "PEREMPUAN", "KAWIN",
                        "BELUM KAWIN", "WNI", "WNA", "SEUMUR HIDUP", "LAHIR", "TEMPAT",
                        "GOL", "DARAH", "JENIS", "KELAMIN", "AGAMA", "STATUS", "PERKAWINAN",
                        "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU", "HINGGA", "ALAMAT", "DESA",
                        "KELURAHAN", "KECAMATAN", "NIK", "NAMA"
                    }
                    is_label_noise = bool(
                        re.search(r'\b(J[EO]N[I1]S|KE[IL1]AM[I1][NM]|GOL|DARAH|AGAMA|STATU|PERKAW|PEKERJ|WARGA|BERLAKU|HINGGA|ALAMAT|PROV|KABUP|KE[LV]DESA|DESA|KECAM)\b', txt)
                        or is_label_text(txt)
                    )
                    is_gender_noise = bool(clean_gender(txt) or re.search(r'LAK|PEREMP|WANITA|PRIA', txt.upper()))
                    is_left_label_column = (b.x_min < (0.15 * w_max) and len(txt) <= 8 and not any(w in _KOTA_KABUPATEN_LEXICON for w in txt.split()))

                    if not is_label_noise and not is_gender_noise and not is_left_label_column and txt not in noise_terms:
                        tokenized_txt = tokenize_name(txt) or txt
                        nama_val = extracted_raw["nama"].get("val") or ""
                        is_same_as_nama = (txt == nama_val or tokenized_txt == nama_val)

                        if not is_same_as_nama:
                            words = txt.split()
                            is_person_name = len(words) > 0 and all(w in _V3_NAME_LEXICON for w in words)

                            # Spatial Y-Band Constraints:
                            # 1. Kel/Desa & Kecamatan must be vertically below RT/RW (if known)
                            is_below_rtrw = (y_rtrw is None) or (b.center_y >= y_rtrw - 8.0)
                            # 2. Kel/Desa & Kecamatan must be vertically above Agama (if known)
                            is_above_agama = (y_agama is None) or (b.center_y <= y_agama + 8.0)

                            if not is_person_name and is_below_rtrw and is_above_agama:
                                kelurahan_candidates.append(b)

        # Disambiguate kelurahan vs kecamatan by Universal Topological Ordering
        if kelurahan_candidates:
            kelurahan_candidates = sorted(kelurahan_candidates, key=lambda b: b.y_min)

            # Skenario 1: Keduanya belum terisi
            if not extracted_raw["kelurahan_desa"]["val"] and not extracted_raw["kecamatan"]["val"]:
                if len(kelurahan_candidates) >= 1:
                    b = kelurahan_candidates[0]
                    c_k = normalize_regional(tokenize_compound_name(clean_text(b.text)))
                    if c_k:
                        extracted_raw["kelurahan_desa"] = {"val": c_k, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
                if len(kelurahan_candidates) >= 2:
                    b = kelurahan_candidates[1]
                    c_kec = normalize_regional(clean_text(b.text))
                    if c_kec:
                        extracted_raw["kecamatan"] = {"val": c_kec, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))

            # Skenario 2: Hanya Kelurahan yang belum terisi
            elif not extracted_raw["kelurahan_desa"]["val"]:
                if len(kelurahan_candidates) >= 1:
                    b = kelurahan_candidates[0]
                    c_k = normalize_regional(tokenize_compound_name(clean_text(b.text)))
                    if c_k:
                        extracted_raw["kelurahan_desa"] = {"val": c_k, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))

            # Skenario 3: Hanya Kecamatan yang belum terisi (Kasus KTP Andri Restu Fauji)
            elif not extracted_raw["kecamatan"]["val"]:
                if len(kelurahan_candidates) >= 1:
                    b = kelurahan_candidates[0]
                    c_kec = normalize_regional(clean_text(b.text))
                    if c_kec:
                        extracted_raw["kecamatan"] = {"val": c_kec, "conf": round(b.confidence, 1)}
                        used_box_ids.add(id(b))
