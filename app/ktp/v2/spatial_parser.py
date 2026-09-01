import re
from typing import List, Dict, Any, Tuple, Optional
from app.ktp.v2.paddle_engine import PaddleTextBox
from app.ktp.v2.field_cleaners import (
    clean_text, clean_nik, clean_date, clean_gender,
    clean_blood_type, clean_marital_status, clean_citizenship,
    clean_rt_rw, tokenize_pekerjaan, normalize_regional, tokenize_compound_name,
    tokenize_name, tokenize_address
)

# Label Anchor Matrix Patterns with OCR Typo Tolerance
LABEL_PATTERNS = {
    "nik": [r'\b(NIK|HIK|MIL|NlK|N1K)\b'],
    "nama": [r'\b(NAMA|NAM)\b'],
    "tempat_tanggal_lahir": [
        r'TEMPA[TL]\s*/?\s*TG[IL1]\s*LAHIR',
        r'TEMPA[TL]\s+LAHIR',
        r'TG[IL1]\s*LAHIR',
        r'\bLAHIR\b',
        r'\bTEMPAT\b'
    ],
    "jenis_kelamin": [
        r'J[EO]N[I1]S\s+KE[IL1]AM[I1][NM]',
        r'JENIS\s+KELAMI[NM]',
        r'\bKE[IL1]AM[I1][NM]\b',
        r'\bJ[EO]N[I1]S\b'
    ],
    "golongan_darah": [r'GOL\.?\s*DARAH', r'\bDARAH\b'],
    "alamat": [r'\b(ALAMAT|ALAMA)\b'],
    "rt_rw": [r'\b(RT\s*/?\s*RW|RT|RW)\b'],
    "kelurahan_desa": [r'KEL\s*/?\s*DESA', r'\bDESA\b', r'\bKELURAHAN\b'],
    "kecamatan": [r'\b(KECAMATAN|KECAM)\b'],
    "agama": [r'\bAGAMA\b'],
    "status_perkawinan": [
        r'STATUS\s+PERKAWINA[NR]',
        r'STATUS\s+PERKAWINAN',
        r'\bPERKAWINA[NR]\b'
    ],
    "pekerjaan": [r'\b(PEKERJAAN|PEKERJA)\b'],
    "kewarganegaraan": [r'\b(KEWARGANEGARAAN|KEWARGAN)\b'],
    "berlaku_hingga": [r'.*RLAKU\s*HINGGA', r'.*HINGGA', r'\b[BN]ERLAKU\b'],
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
    """Extracts value text if label and value are combined in a single box (e.g. 'Status Perkawinan: KAWIN')."""
    if not text:
        return None
    s = text.strip()

    # Split by colon ':'
    if ":" in s:
        parts = s.split(":", 1)
        val = parts[1].strip()
        if val:
            return val

    # If no colon, strip matching label pattern from text
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

class SpatialParserV2:
    def parse_ktp(self, text_boxes: List[PaddleTextBox]) -> Dict[str, Dict[str, Any]]:
        """
        Parses text boxes using 4-Layer Multi-Strategy Pipeline.
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

        def is_signature_zone(b: PaddleTextBox) -> bool:
            return b.x_min > (0.55 * w_max) and b.y_min > (0.55 * h_max)

        for line in lines:
            line_str = " ".join(b.text for b in line)

            line_avg_y = sum(b.center_y for b in line) / len(line)
            if line_avg_y < (0.18 * h_max) and ("PROVINSI" in line_str.upper() or "KABUPATEN" in line_str.upper()):
                continue

            # 1. Compound Line: TEMPAT/TGL LAHIR
            if is_label_text(line_str) == "tempat_tanggal_lahir" or re.search(r'TEMPA[TL].*LAHIR|TG[IL1].*LAHIR', line_str, re.I):
                val_boxes = [b for b in line if not is_label_text(b.text) and not is_signature_zone(b)]
                val_str = ""
                avg_conf = 0.0

                if val_boxes:
                    val_str = " ".join(b.text for b in val_boxes)
                    avg_conf = sum(b.confidence for b in val_boxes) / len(val_boxes)
                else:
                    for b in line:
                        inline_v = extract_inline_value(b.text, "tempat_tanggal_lahir")
                        if inline_v:
                            val_str = inline_v
                            avg_conf = b.confidence
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

                    c_city = clean_text(city_part)
                    c_date = clean_date(date_part)

                    if c_city:
                        extracted_raw["tempat_lahir"] = {"val": c_city, "conf": round(avg_conf, 1)}
                    if c_date:
                        extracted_raw["tanggal_lahir"] = {"val": c_date, "conf": round(avg_conf, 1)}
                continue

            # 2. Compound Line: JENIS KELAMIN & GOL. DARAH
            if is_label_text(line_str) == "jenis_kelamin" or re.search(r'JENIS\s+KELAMI[NM]', line_str, re.I):
                gender_boxes = []
                blood_boxes = []
                found_blood_label = False

                for b in line:
                    if is_signature_zone(b):
                        continue
                    if is_label_text(b.text) == "golongan_darah" or "GOL" in b.text.upper():
                        found_blood_label = True
                        inline_b = extract_inline_value(b.text, "golongan_darah")
                        if inline_b:
                            c_b = clean_blood_type(inline_b)
                            if c_b:
                                extracted_raw["golongan_darah"] = {"val": c_b, "conf": round(b.confidence, 1)}
                        continue
                    if is_label_text(b.text) == "jenis_kelamin":
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

                if blood_boxes and not extracted_raw["golongan_darah"]["val"]:
                    b_str = " ".join(b.text for b in blood_boxes)
                    b_conf = sum(b.confidence for b in blood_boxes) / len(blood_boxes)
                    c_b = clean_blood_type(b_str)
                    if c_b:
                        extracted_raw["golongan_darah"] = {"val": c_b, "conf": round(b_conf, 1)}
                continue

            # 3. Standard Single-Field Labels
            for idx, box in enumerate(line):
                label_key = is_label_text(box.text)
                if not label_key or label_key in ["tempat_tanggal_lahir", "jenis_kelamin", "golongan_darah"]:
                    continue

                # Check inline value FIRST if box contains colon or inline value
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

                if val_str:
                    if label_key == "nik":
                        c_val = clean_nik(val_str)
                    elif label_key == "nama":
                        c_val = tokenize_name(val_str)
                    elif label_key == "rt_rw":
                        c_val = clean_rt_rw(val_str)
                    elif label_key == "agama":
                        c_val = clean_text(val_str)
                    elif label_key == "status_perkawinan":
                        c_val = clean_marital_status(val_str)
                    elif label_key == "kewarganegaraan":
                        c_val = clean_citizenship(val_str)
                    elif label_key in ("kecamatan", "tempat_lahir"):
                        c_val = normalize_regional(clean_text(val_str))
                    elif label_key == "kelurahan_desa":
                        c_val = normalize_regional(tokenize_compound_name(clean_text(val_str)))
                    elif label_key == "pekerjaan":
                        c_val = tokenize_pekerjaan(clean_text(val_str))
                    elif label_key == "berlaku_hingga":
                        raw_clean = clean_text(val_str)
                        # Jika nilai hanya label tanpa value, cari di box kanan
                        if raw_clean in {"HINGGA", "BERLAKU HINGGA", "BERLAKU", "NORLAKU HINGGA"}:
                            raw_clean = None
                            for right_b in line[idx + 1:]:
                                # Tangkap SEUMUR dan SEUNUR (typo M→N pada gambar kualitas rendah)
                                if re.search(r'SEU[MN]UR', right_b.text.upper()):
                                    raw_clean = "SEUMUR HIDUP"
                                    break
                                cd = clean_date(right_b.text)
                                if cd:
                                    raw_clean = cd
                                    break
                        # Normalisasi SEUMUR/SEUNUR variants + strip trailing noise tokens
                        if raw_clean:
                            if re.search(r'SEU[MN]UR', raw_clean):
                                raw_clean = "SEUMUR HIDUP"
                            else:
                                # Jika ada tanggal valid, ambil hanya tanggal (buang noise trailing)
                                date_m = re.search(r'(\d{2}[-./]\d{2}[-./]\d{4})', raw_clean)
                                if date_m:
                                    raw_clean = clean_date(date_m.group(1))
                        c_val = raw_clean
                    elif label_key == "alamat":
                        # Collect multi-line continuation: ambil baris berikutnya
                        # selama baris tersebut tidak dimulai dengan label baru
                        line_idx = lines.index(line)
                        extra_parts = [val_str]
                        extra_conf_sum = avg_conf
                        extra_count = 1
                        for next_line in lines[line_idx + 1:]:
                            next_line_str = " ".join(b.text for b in next_line)
                            # Berhenti jika baris berikutnya mengandung label field baru
                            if any(is_label_text(b.text) for b in next_line):
                                break
                            # Berhenti jika baris terlalu jauh ke bawah (lebih dari 2 baris setelah alamat)
                            next_avg_y = sum(b.center_y for b in next_line) / len(next_line)
                            cur_avg_y = sum(b.center_y for b in line) / len(line)
                            if next_avg_y - cur_avg_y > 80:
                                break
                            non_sig_boxes = [b for b in next_line if not is_signature_zone(b)]
                            if non_sig_boxes:
                                extra_parts.append(" ".join(b.text for b in non_sig_boxes))
                                extra_conf_sum += sum(b.confidence for b in non_sig_boxes) / len(non_sig_boxes)
                                extra_count += 1
                        full_alamat = " ".join(extra_parts)
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
                    break

        # Fallback 2: Positional Structural Fallback (For Label-Less / Cropped Cards like ktp 4)
        if not extracted_raw["nama"]["val"] or not extracted_raw["jenis_kelamin"]["val"]:
            body_boxes = [b for b in text_boxes if b.y_min > (0.18 * h_max) and not is_signature_zone(b)]
            body_boxes = sorted(body_boxes, key=lambda b: b.y_min)

            for b in body_boxes:
                txt = b.text.strip()
                if not txt:
                    continue

                if not extracted_raw["jenis_kelamin"]["val"]:
                    cg = clean_gender(txt)
                    if cg:
                        extracted_raw["jenis_kelamin"] = {"val": cg, "conf": round(b.confidence, 1)}
                        continue

                if not extracted_raw["agama"]["val"]:
                    clean_ag = clean_text(txt)
                    if clean_ag and clean_ag.upper() in ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KHONGHUCU"]:
                        extracted_raw["agama"] = {"val": clean_ag.upper(), "conf": round(b.confidence, 1)}
                        continue

                if not extracted_raw["status_perkawinan"]["val"]:
                    cs = clean_marital_status(txt)
                    if cs:
                        extracted_raw["status_perkawinan"] = {"val": cs, "conf": round(b.confidence, 1)}
                        continue

                if not extracted_raw["kewarganegaraan"]["val"]:
                    ck = clean_citizenship(txt)
                    if ck:
                        extracted_raw["kewarganegaraan"] = {"val": ck, "conf": round(b.confidence, 1)}
                        continue

                if not extracted_raw["pekerjaan"]["val"]:
                    if "BURUH" in txt.upper() or "PELAJAR" in txt.upper() or "SWASTA" in txt.upper() or "PNS" in txt.upper():
                        extracted_raw["pekerjaan"] = {"val": tokenize_pekerjaan(txt), "conf": round(b.confidence, 1)}
                        continue

                if not extracted_raw["berlaku_hingga"]["val"]:
                    cd = clean_date(txt)
                    # Hanya assign jika teks adalah MURNI tanggal atau SEUMUR HIDUP
                    # Tolak jika teks berisi nama kota + tanggal (itu adalah tempat/tgl lahir)
                    is_city_date = bool(re.search(r'[A-Z]{3,}[^\d]*\d{2}[\-\./]\d{2}[\-\./]\d{4}', txt))
                    if "SEUMUR" in txt.upper():
                        extracted_raw["berlaku_hingga"] = {"val": "SEUMUR HIDUP", "conf": round(b.confidence, 1)}
                        continue
                    elif cd and not is_city_date:
                        extracted_raw["berlaku_hingga"] = {"val": cd, "conf": round(b.confidence, 1)}
                        continue

                # Birth place & date check
                if (not extracted_raw["tempat_lahir"]["val"] or not extracted_raw["tanggal_lahir"]["val"]) and clean_date(txt):
                    parts = re.split(r'[,:\.]+', txt)
                    if len(parts) >= 2:
                        c_city = clean_text(parts[0])
                        c_date = clean_date(" ".join(parts[1:]))
                        if c_city:
                            extracted_raw["tempat_lahir"] = {"val": c_city, "conf": round(b.confidence, 1)}
                        if c_date:
                            extracted_raw["tanggal_lahir"] = {"val": c_date, "conf": round(b.confidence, 1)}
                    continue

                # Name fallback (all uppercase text before birth place and after NIK)
                if not extracted_raw["nama"]["val"] and not is_label_text(txt) and len(txt) > 3 and not re.search(r'\d', txt):
                    extracted_raw["nama"] = {"val": tokenize_name(txt), "conf": round(b.confidence, 1)}

        # Layer 5: Pattern-Based Field Guesser (for Label-Less / Cropped Cards)
        self._layer5_pattern_guesser(text_boxes, extracted_raw, h_max, w_max)

        return extracted_raw

    def _layer5_pattern_guesser(
        self,
        text_boxes: List[PaddleTextBox],
        extracted_raw: Dict[str, Dict[str, Any]],
        h_max: float,
        w_max: float
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
            if txt_clean not in assigned_vals and not is_label_text(b.text):
                unassigned.append(b)

        unassigned = sorted(unassigned, key=lambda b: b.y_min)
        kelurahan_candidates: List[PaddleTextBox] = []

        for b in unassigned:
            txt = clean_text(b.text) or ""

            # 1. Jenis kelamin
            if not extracted_raw["jenis_kelamin"]["val"]:
                cg = clean_gender(txt)
                if cg:
                    extracted_raw["jenis_kelamin"] = {"val": cg, "conf": round(b.confidence, 1)}
                    assigned_vals.add(cg)
                    continue

            # 2. RT/RW — format NNN/NNN
            if not extracted_raw["rt_rw"]["val"]:
                clean_rtrw = clean_rt_rw(txt)
                if clean_rtrw and re.match(r'^\d{1,3}/\d{1,3}$', clean_rtrw):
                    extracted_raw["rt_rw"] = {"val": clean_rtrw, "conf": round(b.confidence, 1)}
                    assigned_vals.add(clean_rtrw)
                    continue

            # 3. Alamat — prefix KP., JL., KMP., PERUM., GG., DSN., BLOK
            if not extracted_raw["alamat"]["val"]:
                if re.match(r'^(KP\.|JL\.|KMP\.|PERUM\.|GG\.|DSN\.|BLOK|KP\s|JL\s)', txt, re.I):
                    extracted_raw["alamat"] = {"val": txt, "conf": round(b.confidence, 1)}
                    assigned_vals.add(txt)
                    continue

            # 4. Agama — exact list match
            if not extracted_raw["agama"]["val"]:
                if txt.upper() in ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KHONGHUCU"]:
                    extracted_raw["agama"] = {"val": txt.upper(), "conf": round(b.confidence, 1)}
                    assigned_vals.add(txt.upper())
                    continue

            # 5. Kelurahan/kecamatan candidates — uppercase letters only, len > 3, no digits
            if not extracted_raw["kelurahan_desa"]["val"] or not extracted_raw["kecamatan"]["val"]:
                if re.match(r'^[A-Z\s\.\-]+$', txt) and len(txt) > 3 and not re.search(r'\d', txt):
                    noise_terms = {
                        "PROVINSI", "KABUPATEN", "LAKI-LAKI", "PEREMPUAN", "KAWIN",
                        "BELUM KAWIN", "WNI", "WNA", "SEUMUR HIDUP", "LAHIR", "TEMPAT",
                        "GOL", "DARAH", "JENIS", "KELAMIN", "AGAMA", "STATUS", "PERKAWINAN",
                        "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU", "HINGGA", "ALAMAT", "DESA",
                        "KELURAHAN", "KECAMATAN", "NIK", "NAMA",
                        "JONIS", "KEIAMIN", "JONIS KEIAMIN", "JENIS KELAMIN"
                    }
                    is_label_noise = bool(re.search(r'\b(J[EO]N[I1]S|KE[IL1]AM[I1][NM]|GOL|DARAH|AGAMA|STATU|PERKAW|PEKERJ|WARGA|BERLAKU|HINGGA|ALAMAT|PROV|KABUP)\b', txt))
                    if not is_label_noise and txt not in noise_terms:
                        # Guard: bandingkan raw txt DAN tokenized version terhadap nilai nama
                        # Menangkap kasus DEDENKUSMANI (raw) vs DEDEN KUSMANI (nama tersimpan)
                        from app.ktp.v2.field_cleaners import _V2_NAME_LEXICON, tokenize_name
                        tokenized_txt = tokenize_name(txt) or txt
                        nama_val = extracted_raw["nama"].get("val") or ""
                        is_same_as_nama = (txt == nama_val or tokenized_txt == nama_val)
                        if not is_same_as_nama:
                            words = txt.split()
                            is_person_name = len(words) > 0 and all(w in _V2_NAME_LEXICON for w in words)
                            if not is_person_name:
                                kelurahan_candidates.append(b)

        # Disambiguate kelurahan vs kecamatan by Y-position ordering
        if kelurahan_candidates:
            kelurahan_candidates = sorted(kelurahan_candidates, key=lambda b: b.y_min)
            if not extracted_raw["kelurahan_desa"]["val"] and len(kelurahan_candidates) >= 1:
                b = kelurahan_candidates[0]
                c_k = normalize_regional(tokenize_compound_name(clean_text(b.text)))
                if c_k:
                    extracted_raw["kelurahan_desa"] = {"val": c_k, "conf": round(b.confidence, 1)}
            if not extracted_raw["kecamatan"]["val"] and len(kelurahan_candidates) >= 2:
                b = kelurahan_candidates[1]
                c_kec = normalize_regional(clean_text(b.text))
                if c_kec:
                    extracted_raw["kecamatan"] = {"val": c_kec, "conf": round(b.confidence, 1)}

