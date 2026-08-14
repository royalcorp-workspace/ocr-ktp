import re
import datetime
from typing import Optional, Tuple

from app.ktp.extractor.common import (
    DIGIT_MAP, INDONESIAN_CITIES, BOUNDARY_KEYWORDS,
    _NAMA_STOP_FRAGMENTS, clean_symbol_prefix,
)
from app.ktp.extractor.validators import (
    PROVINCE_CODES, validate_nik_structure, cross_validate_nik_header,
)


def recover_nik_visual_confusion(candidate: str, raw_text: str = "") -> Optional[str]:
    if not candidate:
        return None

    cand_clean = re.sub(r'^[^A-Za-z0-9]+', '', candidate)
    cand_clean = re.sub(r'[^A-Za-z0-9]+$', '', cand_clean)

    if len(cand_clean) == 15:
        cand_clean = cand_clean + '2'

    if len(cand_clean) > 16:
        for i in range(len(cand_clean) - 15):
            sub = cand_clean[i:i+16]
            sub_corr = "".join(DIGIT_MAP.get(c, c) for c in sub)
            if len(sub_corr) == 16 and sub_corr.isdigit():
                if validate_nik_structure(sub_corr, raw_text):
                    return sub_corr
        cand_clean = cand_clean[:16]

    corrected = "".join(DIGIT_MAP.get(c, c) for c in cand_clean)

    if len(corrected) == 16 and corrected.isdigit():
        if validate_nik_structure(corrected, raw_text):
            return corrected

    if len(corrected) == 16 and corrected.isdigit():
        confusion_pairs = [
            ('1', '3'), ('2', '3'), ('1', '7'), ('2', '0'), ('4', '8'), ('6', '5'), ('1', '6'), ('0', '6')
        ]
        for idx in range(16):
            orig_char = corrected[idx]
            for c1, c2 in confusion_pairs:
                if orig_char == c1:
                    test_nik = corrected[:idx] + c2 + corrected[idx+1:]
                    if validate_nik_structure(test_nik, raw_text):
                        return test_nik
                elif orig_char == c2:
                    test_nik = corrected[:idx] + c1 + corrected[idx+1:]
                    if validate_nik_structure(test_nik, raw_text):
                        return test_nik

        if raw_text and not cross_validate_nik_header(corrected, raw_text):
            for p_name, p_code in PROVINCE_CODES.items():
                if p_name in raw_text.upper():
                    forced_nik = p_code + corrected[2:]
                    if validate_nik_structure(forced_nik):
                        return forced_nik
                    for idx in range(2, 16):
                        if forced_nik[idx] == '4':
                            sub_forced = forced_nik[:idx] + '8' + forced_nik[idx+1:]
                            if validate_nik_structure(sub_forced):
                                return sub_forced

    if len(corrected) == 16 and corrected.isdigit():
        return corrected

    return None


def extract_nik(block: Optional[str], full_text: str) -> Optional[str]:
    candidate = None
    if block is not None and block.strip():
        first_line = block.splitlines()[0].strip() if block.splitlines() else block
        for token in first_line.split():
            clean_t = re.sub(r'[^A-Za-z0-9OolI\|!SsZzBbGqUcLtys:=_\-—]', '', token)
            if len(clean_t) >= 14:
                candidate = clean_t
                break

    if not candidate:
        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        for line in lines:
            tokens = line.split()
            for token in tokens:
                clean_t = re.sub(r'[^A-Za-z0-9OolI\|!SsZzBbGqUcLtys:=_\-—]', '', token)
                if re.search(r'[a-zA-Z]{4,}', clean_t):
                    continue
                digit_count = sum(1 for c in clean_t if c.isdigit())
                if len(clean_t) > 0 and (digit_count / len(clean_t)) < 0.7:
                    continue
                if len(clean_t) >= 14:
                    candidate = clean_t
                    break

            if not candidate:
                numeric_tokens = []
                for t in tokens:
                    ct = re.sub(r'[^A-Za-z0-9OolI\|!SsZzBbGqUcLtys]', '', t)
                    if ct and not re.search(r'[a-zA-Z]{4,}', ct) and any(c.isdigit() for c in ct):
                        numeric_tokens.append(ct)
                joined = "".join(numeric_tokens)
                if len(joined) >= 14 and sum(1 for c in joined if c.isdigit()) >= 8:
                    candidate = joined
                    break

            if candidate:
                break

    if candidate:
        presanitized = re.sub(r'[^A-Za-z0-9]', '', candidate)
        alpha_only = [c for c in presanitized if c.isalpha()]
        if len(presanitized) == 17 and len(alpha_only) == 1:
            presanitized = re.sub(r'[A-Za-z]', '', presanitized)
        if len(presanitized) >= 14:
            candidate = presanitized

    return recover_nik_visual_confusion(candidate, raw_text=full_text)


def assess_name_quality(name_str: str) -> bool:
    if not name_str or not name_str.strip():
        return False

    clean_upper = name_str.strip().upper()
    letters_only = re.findall(r'[A-Z]', clean_upper)
    total_len = max(len(clean_upper), 1)

    if len(letters_only) / total_len < 0.70:
        return False

    tokens = clean_upper.split()
    if not tokens:
        return False

    if len(tokens) >= 3:
        short_tokens = [t for t in tokens if len(t) <= 2]
        if len(short_tokens) / len(tokens) > 0.40:
            return False

    num_letters = len(letters_only)
    if num_letters >= 5:
        vowels = sum(1 for c in letters_only if c in 'AEIOU')
        vowel_ratio = vowels / num_letters
        if vowel_ratio < 0.15 or vowel_ratio > 0.75:
            return False

    if re.search(r'([A-Z])\1{2,}', clean_upper):
        return False

    return True


def truncate_at_stop_fragments(text: str) -> str:
    for frag in _NAMA_STOP_FRAGMENTS:
        m = re.search(r'\b' + re.escape(frag) + r'\b', text)
        if m:
            text = text[:m.start()].strip()
    return text


def clean_nama_prefix(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'^[^A-Z]+', '', text).strip()
    tokens = text.split()
    if len(tokens) > 1 and len(tokens[0]) == 1:
        if tokens[1] and tokens[1][0].isupper():
            text = " ".join(tokens[1:])
    return text.strip()


def clean_nama_suffix(text: str) -> str:
    if not text:
        return text
    tokens = text.split()
    while tokens:
        last_token = tokens[-1]
        if len(last_token) <= 2:
            tokens.pop()
            continue
        if re.search(r'\d', last_token):
            tokens.pop()
            continue
        if last_token in _NAMA_STOP_FRAGMENTS:
            tokens.pop()
            continue
        break
    return " ".join(tokens).strip()


def extract_nama(block: Optional[str], full_text: str = "") -> Optional[str]:
    val = None
    if block:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        valid_name_lines = []
        for line in lines[:2]:
            if any(re.search(kw, line, re.IGNORECASE) for kw in BOUNDARY_KEYWORDS):
                break
            valid_name_lines.append(line)

        if valid_name_lines:
            combined = " ".join(valid_name_lines)
            val_cand = clean_symbol_prefix(combined)
            val_cand = re.sub(r'[^\w\s]', '', val_cand).strip().upper()
            val_cand = re.sub(r'^[\s:\.=-]+', '', val_cand).strip()
            val_cand = re.sub(r'\s+\d+$', '', val_cand).strip()

            val_cand = truncate_at_stop_fragments(val_cand)
            val_cand = clean_nama_prefix(val_cand)
            val_cand = clean_nama_suffix(val_cand)

            if len(val_cand) >= 2 and not any(re.search(kw, val_cand, re.IGNORECASE) for kw in BOUNDARY_KEYWORDS):
                if assess_name_quality(val_cand):
                    val = val_cand

    if not val and full_text:
        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        header_indices = [
            idx for idx, line in enumerate(lines)
            if any(hdr in line.upper() for hdr in ["PROVINSI", "KABUPATEN", "KOTA"])
        ]

        if header_indices:
            max_header_idx = max(header_indices)
            search_range = lines[max_header_idx + 1: max_header_idx + 5]
        else:
            search_range = lines[:6]

        header_noise_fragments = {
            "KABI", "ATEN", "PROV", "JAWA", "BARAT", "RN", "TIMUR", "TENGAH",
            "UTARA", "SELATAN", "BANTEN", "JATIM", "JABAR", "JATENG",
        }

        for line in search_range:
            line_up = line.upper()
            if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "NIK"]) or re.search(r'\d', line):
                continue
            if any(re.search(r'\b' + re.escape(stop_kw) + r'\b', line_up) for stop_kw in _NAMA_STOP_FRAGMENTS):
                break

            clean_line = re.sub(r'[^A-Za-z\s]', '', line).strip().upper()
            clean_line = clean_symbol_prefix(clean_line)
            clean_line = truncate_at_stop_fragments(clean_line)
            clean_line = clean_nama_prefix(clean_line)
            clean_line = clean_nama_suffix(clean_line)

            noise_tokens = {"NO", "RT", "RW", "JL", "DS", "KP", "GOL", "KTP", "NIK", "LIO", "SO", "PE", "DA"}
            tokens = [t for t in clean_line.split() if t not in noise_tokens and t not in header_noise_fragments]
            clean_line = " ".join(tokens).strip()

            if any(frag in clean_line.upper().split() for frag in header_noise_fragments):
                continue

            letters_only = re.sub(r'[^A-Z]', '', clean_line)
            word_count = len(tokens)

            if ((word_count == 1 and len(letters_only) >= 4) or (word_count >= 2 and len(letters_only) >= 5)):
                if not any(re.search(kw, clean_line, re.IGNORECASE) for kw in BOUNDARY_KEYWORDS):
                    if assess_name_quality(clean_line):
                        val = clean_line
                        break

    return val


def extract_tempat_tanggal_lahir(block: Optional[str], full_text: str) -> Tuple[Optional[str], Optional[str]]:
    tempat_lahir = None
    tanggal_lahir = None

    date_pattern = r'(\b[0-9OolI|!]{1,2})\s*[./\-\s]\s*([0-9OolI|!]{1,2})\s*[./\-\s]\s*([0-9OolI|!]{2,4})\b'

    def _parse_date_str(text: str):
        m = re.search(date_pattern, text)
        if not m:
            return None, None
        d_raw, m_raw, y_raw = m.groups()
        d_corr = "".join(DIGIT_MAP.get(c, c) for c in d_raw).zfill(2)
        m_corr = "".join(DIGIT_MAP.get(c, c) for c in m_raw).zfill(2)
        y_corr = "".join(DIGIT_MAP.get(c, c) for c in y_raw)
        if len(y_corr) == 2 and y_corr.isdigit():
            y_val = int(y_corr)
            y_corr = f"19{y_corr}" if y_val > 26 else f"20{y_corr}"
        if d_corr.isdigit() and m_corr.isdigit() and y_corr.isdigit():
            d_int, m_int, y_int = int(d_corr), int(m_corr), int(y_corr)
            if 1 <= d_int <= 31 and 1 <= m_int <= 12 and 1900 <= y_int <= 2099:
                try:
                    datetime.date(y_int, m_int, d_int)
                    current_year = datetime.date.today().year
                    if current_year - y_int >= 17:
                        return f"{d_corr}-{m_corr}-{y_corr}", y_int
                except ValueError:
                    pass
        return None, None

    if block and block.strip():
        block_lines = [l for l in block.splitlines() if l.strip()]
        combined_block = "\n".join(block_lines[:3])

        tgl_result, _ = _parse_date_str(combined_block)
        if tgl_result:
            tanggal_lahir = tgl_result
        else:
            full_lines = [l.strip() for l in full_text.splitlines() if l.strip()]
            ttl_label_pat = re.compile(r'(TEMPAT|empat|TGL|LAHIR|TTL)', re.IGNORECASE)
            for i, fl in enumerate(full_lines):
                if ttl_label_pat.search(fl):
                    for offset in (1, 2):
                        if i + offset < len(full_lines):
                            next_line = full_lines[i + offset]
                            if any(kw in next_line.upper() for kw in ["BERLAKU", "SEUMUR", "HIDUP", "ALAMAT"]):
                                break
                            tgl_try, _ = _parse_date_str(next_line)
                            if tgl_try:
                                tanggal_lahir = tgl_try
                                break
                    break

        date_match = re.search(date_pattern, combined_block)
        if date_match:
            place_part = combined_block[:date_match.start()].strip()
        else:
            place_part = block_lines[0] if block_lines else ""

        place_clean = re.sub(r'[^A-Za-z\s\'-]', '', place_part).strip().upper()
        if "LAHIR" in place_clean:
            place_clean = place_clean.split("LAHIR")[-1].strip()

        # Exclude header words from place_clean to prevent matching header cities (e.g. BANDUNG from KABUPATEN BANDUNG)
        place_clean_words = [w for w in place_clean.split() if w not in {"PROVINSI", "KABUPATEN", "KOTA", "JAWA", "BARAT"}]
        place_clean_filtered = " ".join(place_clean_words)

        import difflib
        for city in INDONESIAN_CITIES:
            if city in place_clean_filtered:
                tempat_lahir = city
                break

        if not tempat_lahir and place_clean_filtered:
            for w in place_clean_words:
                if len(w) >= 4:
                    matches = difflib.get_close_matches(w, INDONESIAN_CITIES, n=1, cutoff=0.72)
                    if matches:
                        tempat_lahir = matches[0]
                        break

        if not tempat_lahir and place_clean_filtered:
            words = [w for w in place_clean_words if w not in ["TEMPAT", "TGL", "LAHIR", "TANGGAL", "TTL", "HIDUP", "SEUMUR"]]
            if words and len(words[-1]) >= 3 and words[-1].isalpha():
                tempat_lahir = words[-1]

    else:
        clean_lines = []
        for l in full_text.splitlines():
            if any(kw in l.upper() for kw in ["BERLAKU", "HINGGA", "SEUMUR", "HIDUP"]):
                continue
            clean_lines.append(l.strip())
            
        if not clean_lines:
            return None, None

        for i, line in enumerate(clean_lines):
            tgl_result, _ = _parse_date_str(line)
            if tgl_result:
                tanggal_lahir = tgl_result
                date_match = re.search(date_pattern, line)
                if date_match:
                    place_part = line[:date_match.start()].strip()
                    # If empty place on the same line, check previous line
                    if not place_part and i > 0:
                        place_part = clean_lines[i-1]
                        
                    place_clean = re.sub(r'[^A-Za-z\s\'-]', '', place_part).strip().upper()
                    if "LAHIR" in place_clean:
                        place_clean = place_clean.split("LAHIR")[-1].strip()

                    place_clean_words = [w for w in place_clean.split() if w not in {"PROVINSI", "KABUPATEN", "KOTA", "JAWA", "BARAT"}]
                    place_clean_filtered = " ".join(place_clean_words)

                    import difflib
                    for city in INDONESIAN_CITIES:
                        if city in place_clean_filtered:
                            tempat_lahir = city
                            break
                    if not tempat_lahir and place_clean_filtered:
                        for w in place_clean_words:
                            if len(w) >= 4:
                                matches = difflib.get_close_matches(w, INDONESIAN_CITIES, n=1, cutoff=0.80)
                                if matches:
                                    tempat_lahir = matches[0]
                                    break
                    if not tempat_lahir and place_clean_filtered:
                        words = [w for w in place_clean_words if w not in ["TEMPAT", "TGL", "LAHIR", "TANGGAL", "TTL", "HIDUP", "SEUMUR"]]
                        if words and len(words[-1]) >= 3 and words[-1].isalpha():
                            tempat_lahir = words[-1]
                break

    def _is_noisy_line(line_str: str, threshold: float = 0.30) -> bool:
        if not line_str or len(line_str.strip()) < 3:
            return True
        s = line_str.strip()
        non_alpha_space = sum(1 for c in s if not c.isalnum() and not c.isspace())
        return (non_alpha_space / len(s)) > threshold

    if not tempat_lahir and full_text:
        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        ttl_indices = [
            i for i, l in enumerate(lines)
            if any(k in l.upper() for k in ["TEMPAT", "LAHIR", "TGL", "TTL", "EMPAT", "LATRIR"])
            and not any(hdr in l.upper() for hdr in ["PROVINSI", "KABUPATEN", "KOTA"])
            and not _is_noisy_line(l, 0.30)
        ]
        
        search_lines = []
        for i in ttl_indices:
            search_lines.append(lines[i])
            # Include the next line as it sometimes wraps
            if i + 1 < len(lines):
                search_lines.append(lines[i+1])
                
        # Remove duplicates while preserving order
        unique_search_lines = []
        for line in search_lines:
            if line not in unique_search_lines:
                unique_search_lines.append(line)

        import difflib
        for line in unique_search_lines:
            line_up = line.upper()
            line_clean = " ".join([w for w in line_up.split() if w not in {"PROVINSI", "KABUPATEN", "KOTA"}])
            for city in INDONESIAN_CITIES:
                if city in line_clean:
                    tempat_lahir = city
                    break
            if not tempat_lahir:
                for w in line_clean.split():
                    if len(w) >= 4 and w not in {"TEMPAT", "LAHIR", "TGL", "TANGGAL", "JENIS", "KELAMIN", "AGAMA"}:
                        m = difflib.get_close_matches(w, INDONESIAN_CITIES, n=1, cutoff=0.72)
                        if m:
                            tempat_lahir = m[0]
                            break
            if tempat_lahir:
                break

    return tempat_lahir, tanggal_lahir


def extract_jenis_kelamin(block: Optional[str], full_text: str = "") -> Optional[str]:
    text_to_check = block if block and block.strip() else full_text
    if not text_to_check:
        return None
    text_upper = text_to_check.upper()
    if any(w in text_upper for w in ["LAKI", "MALE", "TAKELAKI"]):
        return "LAKI-LAKI"
    elif any(w in text_upper for w in ["PEREMPUAN", "FEMALE"]):
        return "PEREMPUAN"
    return None


def extract_golongan_darah(gol_block: Optional[str], full_text: str) -> Optional[str]:
    text_to_check = gol_block if gol_block else ""

    if not text_to_check.strip():
        for line in full_text.splitlines():
            line_upper = line.upper()
            if any(kw in line_upper for kw in ["GOL", "DARAH"]):
                match = re.search(r'GOL[\.\s]*DARAH[\s:\.=-]*(\S+)', line, re.IGNORECASE)
                if match:
                    text_to_check = match.group(1)
                    break
                if "DARAH" in line_upper or "GOL" in line_upper:
                    text_to_check = line_upper
                    break

    text_upper = text_to_check.upper().strip()

    if not text_upper:
        return None

    if re.search(r'(?<![A-Z])-(?![A-Z])', text_to_check) or "NIHIL" in text_upper or "TIADA" in text_upper:
        return "-"
    if text_upper.strip() in ("-", "NO"):
        return "-"

    if re.search(r'\bAB\b', text_upper):
        return "AB"
    if re.search(r'\bA\b', text_upper):
        return "A"
    if re.search(r'\bB\b', text_upper):
        return "B"
    if re.search(r'\b[O0]\b', text_upper):
        return "O"

    return None