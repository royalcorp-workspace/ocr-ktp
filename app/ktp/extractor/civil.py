import difflib
import re
import datetime
from typing import Optional

from app.ktp.extractor.common import DIGIT_MAP, extract_text_field

AGAMA_LIST = ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KHONGHUCU", "KEPERCAYAAN"]


import difflib
import re
import datetime
from typing import Optional

from app.ktp.extractor.common import DIGIT_MAP, extract_text_field

AGAMA_LIST = ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KHONGHUCU", "KEPERCAYAAN"]


def extract_agama(block: Optional[str], full_text: str = "") -> Optional[str]:
    # Determine the text to inspect (block has priority; full_text only if an Agama label is found)
    text_to_check = None
    if block and block.strip():
        text_to_check = block
    if not text_to_check and full_text and full_text.strip():
        # Searching full_text lines for explicit Agama keywords or known religion values
        for line in full_text.splitlines():
            line_up = line.upper()
            if any(k in line_up for k in ["AGAMA", "AGAM", "ISLAM", "KRISTEN", "KATHOLIK", "KATOLIK", "HINDU", "BUDDHA", "BUDHA", "KHONGHUCU", "KONGHUCU"]):
                text_to_check = line
                break

    if not text_to_check:
        return None

    text_upper = text_to_check.upper()
    if any(w in text_upper for w in ["ISLAM", "MOSLEM", "MUSLIM", "ESLAM", "ESEAM", "ASLA"]):
        return "ISLAM"
    if any(w in text_upper for w in ["KRISTEN", "CHRISTIAN", "KRISTN"]):
        return "KRISTEN"
    if any(w in text_upper for w in ["KATHOLIK", "CATOLIK", "CATHOLIK", "CATHOLIC", "KATOLIK"]):
        return "KATHOLIK"
    if "HINDU" in text_upper:
        return "HINDU"
    if any(w in text_upper for w in ["BUDDHA", "BUDHA", "BUDDHIST"]):
        return "BUDDHA"
    if any(w in text_upper for w in ["KHONGHUCU", "KONGHUCU"]):
        return "KHONGHUCU"
    if "KEPERCAYAAN" in text_upper:
        return "KEPERCAYAAN"

    # Fuzzy match per word against AGAMA_LIST with lower cutoff 0.72 since this is anchored
    words = [w for w in re.findall(r'[A-Z]{4,}', text_upper) if len(w) >= 4]
    for word in words:
        matches = difflib.get_close_matches(word, AGAMA_LIST, n=1, cutoff=0.72)
        if matches:
            return matches[0]

    return None


def extract_status_perkawinan(block: Optional[str], full_text: str = "") -> Optional[str]:
    text_to_check = None
    if block and block.strip():
        text_to_check = block
    elif full_text and full_text.strip():
        for line in full_text.splitlines():
            if any(k in line.upper() for k in ["STATUS", "PERKAWINAN", "KAWIN", "BELUM", "CERAI"]):
                text_to_check = line
                break

    if not text_to_check:
        return None

    text_upper = text_to_check.upper()
    if any(w in text_upper for w in ["BELUM KAWIN", "BELUMKAWIN", "SINGLE", "UNMARRIED"]) or ("BELUM" in text_upper and "KAWIN" in text_upper):
        return "BELUM KAWIN"
    if "CERAI HIDUP" in text_upper or "DIVORCED" in text_upper:
        return "CERAI HIDUP"
    if "CERAI MATI" in text_upper:
        return "CERAI MATI"
    if "KAWIN" in text_upper or "MARRIED" in text_upper:
        return "KAWIN"

    return None


def extract_pekerjaan(block: Optional[str], full_text: str = "") -> Optional[str]:
    std_jobs = sorted([
        "PELAJAR/MAHASISWA", "PELAJAR / MAHASISWA", "PELAJAR/ MAHASISWA", "PELAJAR /MAHASISWA",
        "KARYAWAN SWASTA", "PEGAWAI SWASTA", "WIRASWASTA",
        "BURUH HARIAN LEPAS", "BURUH",
        "MENGURUS RUMAH TANGGA", "IBU RUMAH TANGGA",
        "PELAJAR", "MAHASISWA",
        "PNS", "TNI", "POLRI",
        "PETANI", "PEDAGANG", "NELAYAN", "PENSIUNAN",
        "SOPIR", "TUKANG", "SATPAM", "SECURITY",
        "GURU", "DOKTER",
        "OTHERS",
    ], key=len, reverse=True)

    if block and block.strip():
        val = extract_text_field(block)
        if val:
            val = re.sub(r'^\d+\s+', '', val).strip()
            dot_match = re.search(r'\.\s+[A-Z]', val)
            if dot_match:
                val = val[:dot_match.start()].strip()

            for std in std_jobs:
                if std in val:
                    if "PELAJAR" in std and "MAHASISWA" in std:
                        return "PELAJAR/MAHASISWA"
                    return std
                    
            # Fallback ke fuzzy match untuk typo (misal "A H HARIAN LEPAS" -> "BURUH HARIAN LEPAS")
            matches = difflib.get_close_matches(val, std_jobs, n=1, cutoff=0.65)
            if matches:
                if "PELAJAR" in matches[0] and "MAHASISWA" in matches[0]:
                    return "PELAJAR/MAHASISWA"
                return matches[0]
                
            return val

    if full_text and full_text.strip():
        for line in full_text.splitlines():
            line_up = line.upper()
            if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "NIK"]):
                continue
            if any(k in line_up for k in ["PEKERJAAN", "Pekerjaar", "ekerjaan"]) or any(std in line_up for std in std_jobs):
                for std in std_jobs:
                    if std in line_up:
                        if "PELAJAR" in std and "MAHASISWA" in std:
                            return "PELAJAR/MAHASISWA"
                        return std

    return None


def extract_kewarganegaraan(block: Optional[str], full_text: str = "") -> Optional[str]:
    text_to_check = None
    if block and block.strip():
        text_to_check = block
    elif full_text and full_text.strip():
        for line in full_text.splitlines():
            if any(k in line.upper() for k in ["KEWARGANEGARAAN", "KWN", "WNI", "WNA"]):
                text_to_check = line
                break

    if not text_to_check:
        return None

    text_upper = text_to_check.upper().strip()
    if any(w in text_upper for w in ["WNI", "WNl", "WN1", "WNL", "VWNI"]):
        return "WNI"
    if "WNA" in text_upper:
        return "WNA"
    if "CHINA" in text_upper:
        return "CHINA"

    return None


def extract_berlaku_hingga(block: Optional[str], full_text: str = "") -> Optional[str]:
    full_str = (full_text or "").upper()
    if "SEUMUR" in full_str or "HIDUP" in full_str:
        return "SEUMUR HIDUP"

    text_to_check = block if (block and block.strip()) else ""
    if not text_to_check and full_text:
        for line in full_text.splitlines():
            if any(k in line.upper() for k in ["BERLAKU", "HINGGA", "SEUMUR"]):
                text_to_check = line
                break

    if not text_to_check:
        return None

    date_pattern = r'(\b[0-9OolI|!]{1,2})\s*[./-]\s*([0-9OolI|!]{1,2})\s*[./-]\s*([0-9OolI|!]{4})\b'
    text_upper = text_to_check.upper()
    if "SEUMUR" in text_upper or "HIDUP" in text_upper:
        return "SEUMUR HIDUP"

    date_match = re.search(date_pattern, text_to_check)
    if date_match:
        d_raw, m_raw, y_raw = date_match.groups()
        d_corr = "".join(DIGIT_MAP.get(c, c) for c in d_raw).zfill(2)
        m_corr = "".join(DIGIT_MAP.get(c, c) for c in m_raw).zfill(2)
        y_corr = "".join(DIGIT_MAP.get(c, c) for c in y_raw)
        if d_corr.isdigit() and m_corr.isdigit() and y_corr.isdigit():
            d_int, m_int, y_int = int(d_corr), int(m_corr), int(y_corr)
            if 1 <= d_int <= 31 and 1 <= m_int <= 12 and 2010 <= y_int <= 2099:
                try:
                    datetime.date(y_int, m_int, d_int)
                    return f"{d_corr}-{m_corr}-{y_corr}"
                except ValueError:
                    pass
    return None