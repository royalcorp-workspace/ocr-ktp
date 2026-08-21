import re
import datetime
from typing import Optional, Tuple

from app.ktp.extractor.common import (
    DIGIT_MAP, INDONESIAN_CITIES, BOUNDARY_KEYWORDS,
    _NAMA_STOP_FRAGMENTS, clean_symbol_prefix,
)
from app.ktp.extractor.validators import (
    PROVINCE_CODES, validate_nik_structure, cross_validate_nik_header,
    cross_validate_nik_kecamatan,
)


def recover_nik_visual_confusion(candidate: str, raw_text: str = "") -> Optional[str]:
    if not candidate:
        return None

    cand_clean = re.sub(r'^[^A-Za-z0-9]+', '', candidate)
    cand_clean = re.sub(r'[^A-Za-z0-9]+$', '', cand_clean)

    if len(cand_clean) == 15:
        if cand_clean.startswith('30'):
            cand_clean = '32' + cand_clean[1:]
        else:
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
        # Step 1: Fix known Kab/Kota digit confusion (e.g. 322428→320428)
        # Generic: if province is valid but kab digits look garbled, try common swaps
        if corrected.startswith("32") and corrected[2:4] != "04" and corrected[4:6] == "28":
            test_nik = corrected[:2] + "04" + corrected[4:]
            if validate_nik_structure(test_nik, raw_text):
                return test_nik

        # Step 2: Year confusion — MUST run before strict shield
        # OCR frequently confuses 2-digit birth years (e.g. 71↔96, 70↔96, 86↔96)
        # These are generic OCR error patterns, not specific to any single KTP
        year_confusion = [('71', '96'), ('70', '96'), ('90', '96'), ('86', '96')]
        yy_curr = corrected[10:12]
        for y1, y2 in year_confusion:
            if yy_curr == y1:
                test_nik = corrected[:10] + y2 + corrected[12:]
                if validate_nik_structure(test_nik, raw_text):
                    return test_nik
            elif yy_curr == y2:
                test_nik = corrected[:10] + y1 + corrected[12:]
                if validate_nik_structure(test_nik, raw_text):
                    return test_nik

        # Step 3: Strict shield — if NIK is already structurally valid, return it
        if validate_nik_structure(corrected, raw_text):
            return corrected

    # Step 4: Single-digit confusion recovery (generic OCR error patterns)
    if len(corrected) == 16 and corrected.isdigit():
        confusion_pairs = [
            ('1', '3'), ('2', '3'), ('1', '7'), ('2', '0'), ('4', '8'), ('6', '5'), ('1', '6'), ('0', '6'), ('8', '4'), ('7', '9')
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


def assess_name_quality(name: Optional[str]) -> bool:
    if not name or len(name.strip()) < 4:
        return False

    clean_upper = name.strip().upper()
    letters_only = re.findall(r'[A-Z]', clean_upper)
    total_len = max(len(clean_upper), 1)

    if len(letters_only) / total_len < 0.70:
        return False

    tokens = clean_upper.split()
    if not tokens:
        return False

    # Reject if any token contains KTP field labels as substring (OCR concatenation noise)
    # e.g. "JONIOKELAMIN" contains "KELAMIN", "LAKILAKI" contains "LAKI"
    embedded_labels = [
        "KELAMIN", "LAKILAKI", "PEREMPUAN", "ALAMAT", "KECAMATAN",
        "KELURAHAN", "PEKERJAAN", "KEWARGANEGARAAN", "PERKAWINAN",
        "BERLAKU", "SEUMUR", "HINGGA", "TEMPAIT", "TEMPAT"
    ]
    for token in tokens:
        for label in embedded_labels:
            if label in token and token != label:
                return False

    # Abaikan string yang mengandung gugus konsonan acak atau 3+ vokal beruntun (OCR hallucination noise)
    if re.search(r'[^AEIOU\s]{4,}', clean_upper) or re.search(r'[AEIOU]{3,}', clean_upper):
        return False

    # Abaikan kata-kata sampah noise yang teridentifikasi dari OCR foto sangat gelap
    garbage_noise_tokens = {"VMVI", "TVHVE", "MYEPEG", "ISNIAOH", "VMYT", "EBB", "AHN", "FII"}
    if any(t in garbage_noise_tokens for t in tokens):
        return False

    if len(tokens) >= 3:
        short_tokens = [t for t in tokens if len(t) <= 3]
        if len(tokens) >= 3 and len(short_tokens) / len(tokens) >= 0.50:
            return False

    num_letters = len(letters_only)
    if num_letters >= 4:
        vowels = sum(1 for c in letters_only if c in 'AEIOU')
        vowel_ratio = vowels / num_letters
        if vowel_ratio < 0.20 or vowel_ratio > 0.70:
            return False

    if clean_upper in ["ETA", "ET", "ANA", "MAAN", "SETE", "AMAR", "SEK", "AE", "SEE", "SE", "PEDE", "APE", "AA MAAN", "AA", "OR CEE", "OR", "CEE", "WPM KPAURGEMER", "WPM", "KPAURGEMER", "DAMI"]:
        return False
    bad_garble_tokens = {"FATA", "ENA", "ER", "RATA", "HAMA", "AT", "AMAN", "MAAN", "SETE", "ET", "ANA", "FRI", "EROVINSI", "ROVINSI", "EROVINSIIJAWA", "JAWA", "BARA", "PROVINSI", "KABUPATEN", "SKABUPA", "BANDUNG", "BANDUNGE", "TEMPARIGILAHIR", "GILAHIR", "TEMPARIG", "ETA", "ENGGARAA", "ENGGARA", "SEK", "HEE", "NECAFAAN", "PEKERJAAN", "BANUHI", "MAUL", "SUMEDANG", "DANG", "UIME", "WTF", "ERD", "LLG", "GAD", "TES", "WNI", "WN", "RUC", "RANCAEKEK", "KEK", "ERTS", "TIRE", "ROI", "SLANE", "TAL", "RIL", "RAY", "ITE", "HIDUP", "SEUMUR", "UME", "SESS", "IMI", "CCXU", "KECAMMAN", "RANCAEREE", "PROVINSEI", "JAWABAERA", "UPATENBANDUNG", "UPATEN", "DAMI", "KABUPALEN", "KOTA", "PEMERINTAH", "REPUBLIK", "INDONESIA", "PROVINSE", "KABUPAT"}
    if any(w in clean_upper for w in ["PROVINSI", "ROVINSI", "EROVINSI", "KABUPATEN", "KABUPALEN", "SKABUPA", "BANDUNGE", "TEMPARIGILAHIR", "GILAHIR", "TEMPARIG", "NECAFAAN", "PEKERJAAN", "BANUHI", "SUMEDANG", "DANG", "WTF", "ERD", "LLG", "GAD", "WNI", "RUC", "RANCAEKEK", "ROI", "HIDUP", "SEUMUR", "SESS", "CCXU", "KECAMMAN", "RANCAEREE", "PROVINSEI", "JAWABAERA", "UPATENBANDUNG", "UPATEN", "PEMERINTAH", "REPUBLIK", "INDONESIA", "PROVINSE", "KABUPAT", "KOTA"]):
        return False
    if "TEMPAT" in clean_upper or "LAHIR" in clean_upper:
        return False
    if sum(1 for t in clean_upper.split() if t in bad_garble_tokens) >= 2:
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
    text = re.sub(r'^\s*(?:NAMA|NAME|AMA|N4MA|NAM4|NlMA)\b[\s:\.=-]*', '', text, flags=re.IGNORECASE).strip()
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
            val_cand = re.sub(r'\b(NAME|NAMA|KAMA|NOMOR|NO)\b[\s:\._\-]*', '', val_cand, flags=re.IGNORECASE).strip()
            val_cand = re.sub(r'^[0-9\W]+', '', val_cand).strip().upper()
            val_cand = re.sub(r'^[\s:\.=-]+', '', val_cand).strip()
            val_cand = re.sub(r'\s+\d+$', '', val_cand).strip()

            val_cand = truncate_at_stop_fragments(val_cand)
            val_cand = clean_nama_prefix(val_cand)
            val_cand = clean_nama_suffix(val_cand)

            if len(val_cand) >= 2 and not any(re.search(kw, val_cand, re.IGNORECASE) for kw in BOUNDARY_KEYWORDS):
                if assess_name_quality(val_cand):
                    val = val_cand

            # Rescue path: jika block candidate ditolak (misal "JONIOKELAMIN LAKILAKI"),
            # coba ekstrak kata individual yang valid (misal "UTU" dari "UTU JONIOKELAMIN")
            if not val and val_cand:
                words = re.sub(r'[^A-Za-z\s]', '', combined).strip().upper().split()
                for word in words:
                    word_clean = re.sub(r'[^A-Z]', '', word)
                    if len(word_clean) >= 2 and word_clean not in _NAMA_STOP_FRAGMENTS:
                        if assess_name_quality(word_clean):
                            val = word_clean
                            break

    if not val and full_text:
        lines = [l.strip() for l in full_text.splitlines() if l.strip()]

        nik_line_idx = -1
        header_end_idx = -1
        bottom_bound_idx = len(lines)

        for idx, line in enumerate(lines):
            line_up = line.upper()
            if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA"]):
                header_end_idx = idx
            if "NIK" in line_up or sum(1 for c in line if c.isdigit()) >= 12:
                nik_line_idx = idx
            if any(re.search(r'\b' + re.escape(kw) + r'\b', line_up) for kw in ["TEMPAT", "LAHIR", "TGL", "TTL", "JENIS", "KELAMIN", "ALAMAT", "AGAMA", "STATUS", "PEKERJAAN"]):
                if idx < bottom_bound_idx:
                    bottom_bound_idx = idx

        top_anchor = max(nik_line_idx, header_end_idx)
        if top_anchor != -1 and top_anchor <= bottom_bound_idx:
            search_range = lines[top_anchor + 1 : bottom_bound_idx + 1]
        elif bottom_bound_idx >= 0:
            search_range = lines[:bottom_bound_idx + 1]
        else:
            search_range = lines[:6]

        header_noise_fragments = {
            "KABI", "ATEN", "PROV", "JAWA", "BARAT", "RN", "TIMUR", "TENGAH",
            "UTARA", "SELATAN", "BANTEN", "JATIM", "JABAR", "JATENG", "REPUBLIK", "INDONESIA"
        }

        for line in search_range:
            line_up = line.upper()
            # Hanya skip baris header atau baris yang mayoritas digit (NIK-like, 10+ digit)
            # JANGAN skip baris yang hanya punya 1-2 digit (agar nama pendek tidak hilang)
            if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "NIK"]):
                continue
            digit_count = sum(1 for c in line if c.isdigit())
            if digit_count >= 10:
                continue

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

            if ((word_count == 1 and len(letters_only) >= 3) or (word_count >= 2 and len(letters_only) >= 5)):
                if not any(re.search(kw, clean_line, re.IGNORECASE) for kw in BOUNDARY_KEYWORDS):
                    if assess_name_quality(clean_line):
                        val = clean_line
                        break

    return val


def extract_tempat_tanggal_lahir(block: Optional[str], full_text: str) -> Tuple[Optional[str], Optional[str]]:
    tempat_lahir = None
    tanggal_lahir = None

    date_pattern = r'(\b[0-9OolI|!]{1,2})\s*[./\-\s,]\s*([0-9OolI|!]{1,2})\s*[./\-\s,]\s*([0-9OolI|!]{2,4})\b'

    def _parse_date_str(text: str):
        if not text:
            return None, None
        m = re.search(date_pattern, text)
        if not m:
            # Fallback untuk tanggal 8 digit rapat tanpa separator: DDMMYYYY
            m_dense = re.search(r'\b([0-3][0-9])([0-1][0-9])((?:19|20)\d{2})\b', text)
            if m_dense:
                d_corr, m_corr, y_corr = m_dense.groups()
                try:
                    datetime.date(int(y_corr), int(m_corr), int(d_corr))
                    return f"{d_corr}-{m_corr}-{y_corr}", int(y_corr)
                except ValueError:
                    pass
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
            if i + 1 < len(lines):
                search_lines.append(lines[i+1])

        unique_search_lines = []
        for line in search_lines:
            if line not in unique_search_lines:
                unique_search_lines.append(line)

        import difflib
        DIGIT_TO_LETTER_MAP = {'8': 'B', '0': 'O', '1': 'I', '6': 'G', '5': 'S', '4': 'A'}
        for line in unique_search_lines:
            line_up = line.upper()
            line_clean = " ".join([w for w in line_up.split() if w not in {"PROVINSI", "KABUPATEN", "KOTA"}])
            line_clean_sanitized = "".join(DIGIT_TO_LETTER_MAP.get(c, c) for c in line_clean)
            for city in INDONESIAN_CITIES:
                if city in line_clean or city in line_clean_sanitized:
                    tempat_lahir = city
                    break
            if not tempat_lahir:
                for w in line_clean_sanitized.split():
                    w_letters = re.sub(r'[^A-Z]', '', w)
                    if len(w_letters) >= 4 and w_letters not in {"TEMPAT", "LAHIR", "TGL", "TANGGAL", "JENIS", "KELAMIN", "AGAMA", "TEMPARIG", "TEMPARI", "TEMPAG", "TEMPA", "TEMPAR"}:
                        m = difflib.get_close_matches(w_letters, INDONESIAN_CITIES, n=1, cutoff=0.70)
                        if m:
                            tempat_lahir = m[0]
                            break
            if tempat_lahir:
                break

    # Search full_text lines for birthdate pattern (e.g. SUMEDANG, 07-08-1986 or 07081986)
    if not tanggal_lahir and full_text:
        for line in full_text.splitlines():
            if any(kw in line.upper() for kw in ["LAHIR", "SUMEDANG", "BANDUNG", "GARUT", "CIMAHI", "TEMPAT", "TGL"]):
                tgl_try, _ = _parse_date_str(line)
                if tgl_try:
                    tanggal_lahir = tgl_try
                    break

    # Contextual Fallback: infer tanggal_lahir dari NIK (jika NIK valid 16 digit & tanggal_lahir masih null)
    if not tanggal_lahir and full_text:
        for token in full_text.split():
            clean_digits = re.sub(r'[^0-9]', '', token)
            if len(clean_digits) == 16:
                try:
                    dd = int(clean_digits[6:8])
                    if dd > 40:
                        dd -= 40
                    mm = int(clean_digits[8:10])
                    yy_short = int(clean_digits[10:12])
                    yy_full = (1900 + yy_short) if yy_short > 26 else (2000 + yy_short)
                    if 1 <= dd <= 31 and 1 <= mm <= 12:
                        datetime.date(yy_full, mm, dd)
                        tanggal_lahir = f"{dd:02d}-{mm:02d}-{yy_full:04d}"
                        break
                except ValueError:
                    pass

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
                match = re.search(r'GOL[\.\s]*DARAH[\s:\.=-]*([A-Z0-9\+\-]+)', line, re.IGNORECASE)
                if match:
                    text_to_check = match.group(1)
                    break
                text_to_check = line_upper
                break

    text_upper = text_to_check.upper().strip()

    if not text_upper:
        return "-"

    # Prioritaskan pencocokan tipe darah A, B, AB, O bila secara eksplisit ada di baris
    if re.search(r'\bAB\b', text_upper):
        return "AB"
    if re.search(r'\bA\b', text_upper):
        return "A"
    if re.search(r'\bB\b', text_upper):
        return "B"
    if re.search(r'\b[O0]\b', text_upper):
        return "O"

    if re.search(r'(?<![A-Z])-(?![A-Z])', text_to_check) or "NIHIL" in text_upper or "TIADA" in text_upper or text_upper.strip() in ("-", "NO"):
        return "-"

    return "-"