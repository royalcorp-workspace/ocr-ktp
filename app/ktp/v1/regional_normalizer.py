import re
import difflib
from typing import Optional

# Daftar kamus wilayah (Kecamatan, Kelurahan/Desa, Kota/Kabupaten umum di Indonesia)
REGIONAL_DICTIONARY = [
    "LINGGARJATI", "JELEGONG", "JATISARI", "SUKAMULYA", "RANCAEKEK WETAN", "RANCAEKEK KENCANA", "RANCAEKEK KULON", "BOJONGSALAM", "BOJONGMANIK",
    "RANCAEKEK", "CICALENGKA", "CIPARAY", "MAJALAYA", "SOLOKANJERUK", "PASEH", "IBUN",
    "KATAPANG", "SOREANG", "MARGAASIH", "MARGAHAYU", "DAYEUHKOLOT", "BALEENDAH", "BANJARAN",
    "CIMAUNG", "PANGALENGAN", "KERTASARI", "PASIRWANGI", "CIMAHI", "BANDUNG", "KOTA BANDUNG",
    "KABUPATEN BANDUNG", "BANDUNG BARAT", "SUMEDANG", "GARUT", "TASIKMALAYA", "CIAMIS",
    "CIMAHI UTARA", "CIMAHI TENGAH", "CIMAHI SELATAN", "COBLONG", "CICENDO", "SUKAJADI",
    "SUKASARI", "ANDIR", "REGOL", "ASTANAANYAR", "LENGKONG", "BATUNUNGGAL", "BUAHBATU",
    "CIBIRU", "PANYILEUKAN", "GEDEBAGE", "RANCASARI", "ARCAMANIK", "MANDALAJATI", "ANTAPANI"
]


def fix_common_ocr_typos(text: str) -> str:
    if not text:
        return text

    res = text.strip()

    # 1. Perbaikan Spasi & Blok Alamat / Kecamatan
    res = re.sub(r'^(KECAMATAN|KECAMMAN|KECAMATN|KECAMAT|AMAIAN)[\s:\.=-]+', '', res, flags=re.IGNORECASE).strip()
    res = re.sub(r'\bPEAMAI\b', 'PERMAI', res, flags=re.IGNORECASE)
    res = re.sub(r'\bPERMAIBLOK\b', 'PERMAI BLOK', res, flags=re.IGNORECASE)
    res = re.sub(r'\bPEAMAIBLOK\b', 'PERMAI BLOK', res, flags=re.IGNORECASE)
    res = re.sub(r'\bBLOK([A-Z0-9]+)\b', r'BLOK \1', res, flags=re.IGNORECASE)
    res = re.sub(r'\bNO([0-9]+)\b', r'NO. \1', res, flags=re.IGNORECASE)

    # Clean double spaces
    res = re.sub(r'\s+', ' ', res).strip()

    return res


def normalize_regional_text(text: Optional[str], field_name: str = "") -> Optional[str]:
    """
    Normalisasi teks wilayah (Kecamatan, Kelurahan/Desa, Tempat Lahir) menggunakan 
    fuzzy matching kamus wilayah untuk memperbaiki karakter awal yang terdistorsi batik KTP
    (misal: 'AANCAEKEK' -> 'RANCAEKEK', 'AANCAEKEK WETAN' -> 'RANCAEKEK WETAN').
    """
    if not text or len(text.strip()) < 3:
        return text

    clean_text = fix_common_ocr_typos(text).upper()
    clean_text = re.sub(r'[\s\-—–_=\+~]+[A-Z0-9]{1,3}$', '', clean_text).strip()

    # Reject false misread headers & garbage lines (misal JENIS KELAMIN ter-misread sebagai LENIS / FANE KELAMIN)
    if any(bad in clean_text for bad in ["KELAMIN", "JENIS", "TAKILAKI", "PEREMPUAN", "LAN. AK", "JERA", "JORA"]):
        return None
    if clean_text in ["LENIS", "LAK1", "TAKILAKI"]:
        return None

    # 1. Instant O(1) exact match check
    if clean_text in REGIONAL_DICTIONARY:
        return clean_text

    # 2. Pre-filter dictionary by string length (+/- 3 chars) to prevent O(N) diff scans
    filtered_cand = [w for w in REGIONAL_DICTIONARY if abs(len(w) - len(clean_text)) <= 3]
    if filtered_cand:
        matches = difflib.get_close_matches(clean_text, filtered_cand, n=1, cutoff=0.75)
        if matches:
            return matches[0]

    return clean_text
