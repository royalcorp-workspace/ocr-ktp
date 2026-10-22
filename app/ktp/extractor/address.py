import re
from typing import Optional

from app.ktp.extractor.common import DIGIT_MAP, clean_symbol_prefix, sanitize_block_text


def extract_alamat(block: Optional[str], full_text: str = "") -> Optional[str]:
    text_to_process = block
    if not text_to_process and full_text:
        for line in full_text.splitlines():
            line_up = line.upper()
            if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "NAMA", "JENIS", "KELAMIN", "AGAMA", "STATUS", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU", "TEMPAT"]):
                continue
            if any(re.search(r'\b' + kw + r'\b', line_up) for kw in ["KP", "JL", "JALAN", "GANG", "GG", "BLOK", "PERUM", "DUSUN", "KAMPUNG", "BOJONG", "DS", "DESA", "KEL", "RT", "RW"]):
                text_to_process = line
                break

    if not text_to_process:
        return None

    sanitized = sanitize_block_text(text_to_process, max_lines=2)
    if not sanitized:
        return None

    val = clean_symbol_prefix(sanitized)
    val = val.upper()
    val = re.sub(r'^(ALAMAT|ALAMA|ALAMAL|ALAMAI|ALAMTI|ALMT|ALAMT|LAMAT|MAMAT|AMAT)[\s:\.=\-+]+', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'^\b(AMAR|AMAR ——|AMAR —|SEK|ENGGARAA|RN)\b[\s:\.=\-—–]*', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'^[\s:\.=\-—–\u2014\u2013]+', '', val).strip()
    val = re.sub(r'\bKP\s*II\b', 'KP.', val)
    val = re.sub(r'\bGCARIAT\b', 'LINGGARJATI', val)
    val = re.sub(r'\s*EN\s+SHEAE.*$', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'\s*\[.*$', '', val).strip()
    val = re.sub(r'\bKPR([A-Z])', r'KP \1', val)
    val = re.sub(r'\bKP\.([A-Z])', r'KP. \1', val)
    val = re.sub(r'\bJA\s+TISARI\b', 'JATISARI', val)
    val = re.sub(r'\bKP\s+JATISARI\b', 'KP. JATISARI', val)
    val = re.sub(r'\bJATISA\b', 'JATISARI', val)
    val = re.sub(r'\bLINGGARJATE\b', 'LINGGARJATI', val)
    val = re.sub(r'\s*[_=]+\s*', ' ', val)
    val = re.sub(r'\s+-\s+', ' ', val)
    val = re.sub(r'\s{2,}', ' ', val).strip()

    val = re.sub(r'\s+\b(DAI|DA|OAI|OA|DAL|DAl|DI|AD|PP\.\s*WS)\b[\.\s]*$', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'^[I|1]\s+(JL)', r'\1', val).strip()
    # Hapus trailing noise symbols dan em-dash fragments (e.g. "» LI", "| A", "— I", "~ SE", "BLOK A-S :")
    val = re.sub(r'[\s\u00bb|~=—–\.\*\#\:\!]+[A-Za-z]{1,2}$', '', val).strip()
    val = re.sub(r'[\s\-—–_=\+~]+$', '', val).strip()
    val = re.sub(r'[\s\-»"“”\'«=_\.\*#|~:\u2014\u2013]+$', '', val).strip()
    val = re.sub(r'\bJATISA\b', 'JATISARI', val).strip()
    val = re.sub(r'[.:\-–\s]+$', '', val).strip()

    return val if val else None


def parse_rt_rw_with_spillover(raw_str: str) -> tuple[Optional[str], Optional[str]]:
    """
    Ekstrak pola RT/RW numerik 3-digit closed set (001/002) dan isolasi teks spillover alfabet.
    Returns:
        (rt_rw, spillover)
    """
    if not raw_str or not raw_str.strip():
        return None, None

    # Hapus format tanggal lahir agar tidak disalahartikan sebagai RT/RW
    date_pattern = r'\b\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b'
    cleaned_input = re.sub(date_pattern, ' ', raw_str)

    # 1. Regex untuk RT/RW pattern (1-3 digit, dipisahkan slash atau variasi separator seperti /RW.)
    match = re.search(r'([0-9OolI|!sSZzBbGqUcD]{1,3})\s*(?:/\s*(?:RW[.:\s]*)?|[/|\\])\s*([0-9OolI|!sSZzBbGqUcD]{1,3})', cleaned_input, re.IGNORECASE)
    matched_span = None
    rt_norm = None
    rw_norm = None

    if match:
        rt_raw, rw_raw = match.groups()
        if any(c.isdigit() for c in rt_raw + rw_raw):
            rt_corr = "".join(DIGIT_MAP.get(c, c) for c in rt_raw)
            rw_corr = "".join(DIGIT_MAP.get(c, c) for c in rw_raw)
            if len(rt_corr) == 3 and rt_corr.startswith("8"):
                rt_corr = "0" + rt_corr[1:]
            if len(rw_corr) == 3 and rw_corr.startswith("8"):
                rw_corr = "0" + rw_corr[1:]

            if rt_corr.isdigit() and rw_corr.isdigit():
                rt_norm = f"{int(rt_corr):03d}"
                rw_norm = f"{int(rw_corr):03d}"
                matched_span = match.span()


    if not rt_norm:
        # Fallback match 6-digit: RT/RW 001 002
        match_6d = re.search(r'\b(?:RT|RW|RT/RW|RTRW|RTIRW|RT/AW)?\s*[:=.\-]?\s*([0-9OolI|!sSZzBbGqUcD]{3})\s*[/:\.\- ]\s*([0-9OolI|!sSZzBbGqUcD]{3})\b', cleaned_input, re.IGNORECASE)
        if match_6d:
            rt_raw, rw_raw = match_6d.groups()
            rt_corr = "".join(DIGIT_MAP.get(c, c) for c in rt_raw)
            rw_corr = "".join(DIGIT_MAP.get(c, c) for c in rw_raw)
            if rt_corr.isdigit() and rw_corr.isdigit():
                rt_norm = f"{int(rt_corr):03d}"
                rw_norm = f"{int(rw_corr):03d}"
                matched_span = match_6d.span()

    if not rt_norm or not rw_norm:
        return None, None

    rt_rw_res = f"{rt_norm}/{rw_norm}"

    # 2. Extract Residual Spillover Text
    if matched_span:
        start_idx, end_idx = matched_span
        residual = cleaned_input[:start_idx] + " " + cleaned_input[end_idx:]
    else:
        residual = ""

    # Bersihkan noise label RT/RW dari residual (preserve dots inside abbreviations like KP. or JL.)
    residual = re.sub(r'\b(RT|RW|RT/RW|RTRW|RTIRW|RT/AW|RT/RAW|RT/RN|ATAW|T/RW|T/AW)\b', ' ', residual, flags=re.IGNORECASE)
    residual = residual.strip(" :.=-—–_\\/")
    residual = re.sub(r'\s+', ' ', residual).strip()

    # Validasi kelayakan spillover sebagai potongan alamat (minimal 3 karakter alfabet)
    alpha_chars = re.sub(r'[^A-Za-z]', '', residual)
    spillover = residual if len(alpha_chars) >= 3 else None

    return rt_rw_res, spillover



def extract_rt_rw_with_spillover(
    block: Optional[str],
    full_text: str = "",
    block_alamat: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    text_to_search = block if block and block.strip() else ""

    if not text_to_search and block_alamat:
        rt_pattern = r'\b(RT|RW|RT/RW|RTRW|RTIRW|RT/AW|RT/RAW|RT/RN)\b'
        for line in block_alamat.splitlines():
            if re.search(rt_pattern, line, re.IGNORECASE):
                text_to_search = block_alamat
                break

    if not text_to_search:
        return None, None

    return parse_rt_rw_with_spillover(text_to_search)


def extract_rt_rw(
    block: Optional[str],
    full_text: str = "",
    block_alamat: Optional[str] = None
) -> Optional[str]:
    rt_rw, _ = extract_rt_rw_with_spillover(block, full_text, block_alamat)
    return rt_rw