import difflib
from typing import Optional

from app.ktp.extractor.common import INDONESIAN_CITIES

PROVINCE_CODES = {
    "ACEH": "11",
    "SUMATERA UTARA": "12", "SUMUT": "12",
    "SUMATERA BARAT": "13", "SUMBAR": "13",
    "RIAU": "14",
    "JAMBI": "15",
    "SUMATERA SELATAN": "16", "SUMSEL": "16",
    "BENGKULU": "17",
    "LAMPUNG": "18",
    "KEPULAUAN BANGKA BELITUNG": "19", "BANGKA BELITUNG": "19",
    "KEPULAUAN RIAU": "21", "KEPRI": "21",
    "DKI JAKARTA": "31", "JAKARTA": "31",
    "JAWA BARAT": "32", "JABAR": "32",
    "JAWA TENGAH": "33", "JATENG": "33",
    "DI YOGYAKARTA": "34", "YOGYAKARTA": "34", "DIY": "34",
    "JAWA TIMUR": "35", "JATIM": "35",
    "BANTEN": "36",
    "BALI": "51",
    "NUSA TENGGARA BARAT": "52", "NTB": "52",
    "NUSA TENGGARA TIMUR": "53", "NTT": "53",
    "KALIMANTAN BARAT": "61", "KALBAR": "61",
    "KALIMANTAN TENGAH": "62", "KALTENG": "62",
    "KALIMANTAN SELATAN": "63", "KALSEL": "63",
    "KALIMANTAN TIMUR": "64", "KALTIM": "64",
    "KALIMANTAN UTARA": "65", "KALTARA": "65",
    "SULAWESI UTARA": "71", "SULUT": "71",
    "SULAWESI TENGAH": "72", "SULTENG": "72",
    "SULAWESI SELATAN": "73", "SULSEL": "73",
    "SULAWESI TENGGARA": "74", "SULTRA": "74",
    "GORONTALO": "75",
    "SULAWESI BARAT": "76", "SULBAR": "76",
    "MALUKU": "81",
    "MALUKU UTARA": "82",
    "PAPUA": "91",
    "PAPUA BARAT": "92",
}


def cross_validate_nik_header(nik: str, raw_text: str) -> bool:
    if not nik or len(nik) < 2 or not raw_text:
        return True

    text_upper = raw_text.upper()
    prov_header = None
    for line in text_upper.splitlines():
        if "PROVINSI" in line or "PROVINS" in line or "PROV" in line:
            prov_header = line
            break

    if not prov_header:
        return True

    detected_code = None
    for p_name, p_code in PROVINCE_CODES.items():
        if p_name in prov_header:
            detected_code = p_code
            break

    if not detected_code:
        return True

    return nik[:2] == detected_code


def validate_nik_structure(nik: str, raw_text: str = "") -> bool:
    if not nik or len(nik) != 16 or not nik.isdigit():
        return False
    try:
        prov = int(nik[0:2])
        kab = int(nik[2:4])
        kec = int(nik[4:6])
        dd = int(nik[6:8])
        mm = int(nik[8:10])
        seq = int(nik[12:16])

        is_valid_prov = (11 <= prov <= 92)
        is_valid_kab = (1 <= kab <= 78)
        is_valid_kec = (1 <= kec <= 75)
        is_valid_dd = (1 <= dd <= 31) or (41 <= dd <= 71)
        is_valid_mm = (1 <= mm <= 12)
        is_valid_seq = (1 <= seq <= 9999)

        struct_valid = is_valid_prov and is_valid_kab and is_valid_kec and is_valid_dd and is_valid_mm and is_valid_seq
        if not struct_valid:
            return False

        if raw_text:
            return cross_validate_nik_header(nik, raw_text)

        return True
    except ValueError:
        return False


def sync_nik_with_birthdate(
    nik: Optional[str],
    tanggal_lahir_str: Optional[str],
    jenis_kelamin_str: Optional[str]
) -> Optional[str]:
    """
    Koreksi cerdas NIK berbasis Tanggal Lahir (DD-MM-YYYY) dan Gender.
    Tanggal lahir yang valid dianggap sebagai Ground Truth kontekstual untuk
    memperbaiki 6-digit tengah NIK (DDMMYY).
    """
    if not nik or len(nik) != 16 or not nik.isdigit():
        return nik
    if not tanggal_lahir_str:
        return nik

    parts = tanggal_lahir_str.split('-')
    if len(parts) != 3 or len(parts[0]) != 2 or len(parts[1]) != 2 or len(parts[2]) != 4:
        return nik

    try:
        d_target = int(parts[0])
        m_target = int(parts[1])
        y_target = parts[2][2:]
    except ValueError:
        return nik

    try:
        actual_dd = int(nik[6:8])
    except ValueError:
        return nik

    # Evaluasi opsi gender (Perempuan = DD + 40, Laki-Laki = DD)
    candidate_targets = []
    if jenis_kelamin_str == "PEREMPUAN":
        candidate_targets.append(f"{(d_target + 40):02d}{m_target:02d}{y_target}")
    elif jenis_kelamin_str == "LAKI-LAKI":
        candidate_targets.append(f"{d_target:02d}{m_target:02d}{y_target}")
    else:
        # Jika gender tidak pasti, uji sesuai kecenderungan DD NIK saat ini
        if actual_dd > 40:
            candidate_targets.append(f"{(d_target + 40):02d}{m_target:02d}{y_target}")
            candidate_targets.append(f"{d_target:02d}{m_target:02d}{y_target}")
        else:
            candidate_targets.append(f"{d_target:02d}{m_target:02d}{y_target}")
            candidate_targets.append(f"{(d_target + 40):02d}{m_target:02d}{y_target}")

    actual_ddmmyy = nik[6:12]

    # Pasangan karakter confusion OCR visual yang umum
    valid_confusion_pairs = {
        ('5', '6'), ('6', '5'), ('0', '6'), ('6', '0'),
        ('1', '7'), ('7', '1'), ('3', '8'), ('8', '3'),
        ('0', '8'), ('8', '0'), ('1', '9'), ('9', '1'),
        ('0', '5'), ('5', '0'), ('0', '1'), ('1', '0'),
        ('1', '4'), ('4', '1'), ('2', '3'), ('3', '2'),
        ('4', '9'), ('9', '4'), ('5', '9'), ('9', '5'),
        ('0', '9'), ('9', '0'), ('1', '3'), ('3', '1'),
        ('2', '8'), ('8', '2'), ('1', '6'), ('6', '1'),
    }

    nik_is_currently_valid = validate_nik_structure(nik)

    for target_ddmmyy in candidate_targets:
        if actual_ddmmyy == target_ddmmyy:
            return nik

        candidate_nik = nik[:6] + target_ddmmyy + nik[12:]
        if not validate_nik_structure(candidate_nik):
            continue

        diff_indices = [i for i, (a, b) in enumerate(zip(actual_ddmmyy, target_ddmmyy)) if a != b]
        diff_count = len(diff_indices)

        # 1. Jika NIK asli tidak valid strukturnya (karena OCR merusak digit tanggal), perbaiki langsung!
        if not nik_is_currently_valid:
            return candidate_nik

        # 2. Jika 1-3 digit berbeda dan memenuhi confusion pair / batasan wajar
        if 1 <= diff_count <= 3:
            is_valid_confusion = all(
                (actual_ddmmyy[idx], target_ddmmyy[idx]) in valid_confusion_pairs
                for idx in diff_indices
            )
            if is_valid_confusion or diff_count <= 2:
                return candidate_nik

    return nik


def correct_tempat_lahir_fuzzy(tempat_lahir: Optional[str]) -> Optional[str]:
    if not tempat_lahir or len(tempat_lahir) < 3:
        return tempat_lahir

    clean_tp = tempat_lahir.upper().strip()

    if clean_tp in INDONESIAN_CITIES:
        return clean_tp

    matches = difflib.get_close_matches(clean_tp, INDONESIAN_CITIES, n=1, cutoff=0.80)
    if matches:
        return matches[0]

    return tempat_lahir