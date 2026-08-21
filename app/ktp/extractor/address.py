import re
from typing import Optional

from app.ktp.extractor.common import DIGIT_MAP, clean_symbol_prefix, sanitize_block_text


def extract_alamat(block: Optional[str], full_text: str = "") -> Optional[str]:
    text_to_process = block
    if not text_to_process and full_text:
        for line in full_text.splitlines():
            line_up = line.upper()
            if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "NAMA"]):
                continue
            if any(kw in line_up for kw in ["KP", "JL", "JALAN", "GANG", "GG", "BLOK", "PERUM", "DUSUN", "KAMPUNG"]):
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

    for kw in [
        r'\b(RT|RW|RT/RW|RTRW|RTIRW|RT/AW|RT/RAW|RT/RN|AT/AW|AT/RW)\b',
        r'\bKEL\b', r'\bDESA\b', r'\bKECAMATAN\b',
        r'\bLAKI\b', r'\bPEREMPUAN\b', r'\bISLAM\b', r'\bKRISTEN\b', r'\bKATHOLIK\b',
        r'\bKAWIN\b', r'\bBELUM\b', r'\bPEKERJAAN\b', r'\bBERLAKU\b'
    ]:
        m = re.search(kw, val, re.IGNORECASE)
        if m:
            val = val[:m.start()].strip()
            break

    val = re.sub(r'[\s~]*RTA?[\s:]+.*$', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'\s+\b(DAI|DA|OAI|OA|DAL|DAl|DI|AD|PP\.\s*WS)\b[\.\s]*$', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'^[I|1]\s+(JL)', r'\1', val).strip()
    # Hapus trailing noise symbols dan em-dash fragments (e.g. "» LI", "| A", "— I", "~ SE", "BLOK A-S :")
    val = re.sub(r'[\s\u00bb|~=—–\.\*\#\:\!]+[A-Za-z]{1,2}$', '', val).strip()
    val = re.sub(r'[\s\-—–_=\+~]+$', '', val).strip()
    val = re.sub(r'[\s\-»"“”\'«=_\.\*#|~:\u2014\u2013]+$', '', val).strip()
    val = re.sub(r'\bJATISA\b', 'JATISARI', val).strip()
    val = re.sub(r'[.:\-–\s]+$', '', val).strip()

    return val if val else None


def extract_rt_rw(
    block: Optional[str],
    full_text: str = "",
    block_alamat: Optional[str] = None
) -> Optional[str]:
    # We only want to search if there's an explicit RT/RW label.
    # The block passed here comes from parse_label_blocks which matched the label.
    text_to_search = block if block and block.strip() else ""

    # If no RT_RW block, we can check block_alamat, but ONLY if we find a strong RT/RW label indicator
    if not text_to_search and block_alamat:
        rt_pattern = r'\b(RT|RW|RT/RW|RTRW|RTIRW|RT/AW|RT/RAW|RT/RN)\b'
        for line in block_alamat.splitlines():
            if re.search(rt_pattern, line, re.IGNORECASE):
                text_to_search = block_alamat
                break

    if not text_to_search:
        return None

    # Remove dates from text_to_search to prevent fabricating RT/RW from birth dates
    date_pattern = r'\b\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b'
    text_to_search = re.sub(date_pattern, ' ', text_to_search)

    match = re.search(r'([0-9OolI|!sSZzBbGqUcD]{1,3})\s*[/|\\7\-_s\s|]\s*([0-9OolI|!sSZzBbGqUcD]{1,3})', text_to_search)
    if match:
        rt_raw, rw_raw = match.groups()
        rt_corr = "".join(DIGIT_MAP.get(c, c) for c in rt_raw)
        rw_corr = "".join(DIGIT_MAP.get(c, c) for c in rw_raw)
        if len(rt_corr) == 3 and rt_corr.startswith("8"):
            rt_corr = "0" + rt_corr[1:]
        if len(rw_corr) == 3 and rw_corr.startswith("8"):
            rw_corr = "0" + rw_corr[1:]

        rt_corr = rt_corr.zfill(3)
        rw_corr = rw_corr.zfill(3)
        if rt_corr.isdigit() and rw_corr.isdigit() and len(rt_corr) <= 3 and len(rw_corr) <= 3:
            return f"{rt_corr}/{rw_corr}"

    match_6digit = re.search(r'\b([0-9OolI|!sSZzBbGqUcD]{3})\s*([0-9OolI|!sSZzBbGqUcD]{3})\b', text_to_search)
    if match_6digit:
        rt_raw, rw_raw = match_6digit.groups()
        rt_corr = "".join(DIGIT_MAP.get(c, c) for c in rt_raw).zfill(3)
        rw_corr = "".join(DIGIT_MAP.get(c, c) for c in rw_raw).zfill(3)
        if rt_corr.isdigit() and rw_corr.isdigit():
            return f"{rt_corr}/{rw_corr}"

    return None