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
    detected_code = None
    for line in text_upper.splitlines():
        if any(hdr in line for hdr in ["PROVINSI", "PROVINS", "PROV", "KABUPATEN", "KOTA"]):
            for p_name, p_code in PROVINCE_CODES.items():
                if p_name in line:
                    detected_code = p_code
                    break
            if detected_code:
                break

    if not detected_code:
        if any(kw in text_upper for kw in ["BANDUNG", "RANCAEKEK", "CIMAHI", "JAWA BARAT", "JABAR"]):
            detected_code = "32"

    if not detected_code:
        return True

    return nik[:2] == detected_code


def cross_validate_nik_kecamatan(nik: str, raw_text: str) -> str:
    """
    Strict NIK Shield Policy: Jika NIK 16 digit sudah valid strukturnya (Provinsi, Kab/Kota, Kecamatan, DOB, Seq),
    JANGAN PERNAH meng-override digit NIK tersebut.
    """
    if not nik or len(nik) != 16 or not nik.isdigit():
        return nik

    # Jika NIK asli hasil OCR sudah valid strukturnya, pertahankan NIK asli!
    if validate_nik_structure(nik):
        return nik

    return nik


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

        if diff_count <= 3:
            all_valid_pairs = all((actual_ddmmyy[i], target_ddmmyy[i]) in valid_confusion_pairs for i in diff_indices)
            if all_valid_pairs and validate_nik_structure(candidate_nik):
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


def vote_nik_character_level(
    base_nik: str, 
    raw_texts: list[str], 
    tanggal_lahir: str | None = None, 
    jenis_kelamin: str | None = None
) -> str:
    """Melakukan Character-Level Consensus pada NIK digit 4 & 5 berdasarkan list raw_text kandidat."""
    if not base_nik or len(base_nik) != 16:
        return base_nik

    from app.ktp.extractor.identity import extract_nik
    nik_candidates = []
    for raw in raw_texts:
        if raw and raw.strip():
            cand_nik = extract_nik(None, raw)
            if cand_nik and len(cand_nik) == 16 and cand_nik.isdigit():
                nik_candidates.append(cand_nik)

    if len(nik_candidates) >= 1:
        voted_chars = list(base_nik)
        for idx in range(16):
            char_weights = {}
            for rank, n_str in enumerate(nik_candidates):
                if len(n_str) == 16:
                    c = n_str[idx]
                    weight = 1.0 - (rank * 0.01)
                    char_weights[c] = char_weights.get(c, 0.0) + weight
            
            if char_weights:
                voted_chars[idx] = max(char_weights.keys(), key=lambda k: char_weights[k])
            
        voted_nik = "".join(voted_chars)

        if tanggal_lahir:
            voted_nik = sync_nik_with_birthdate(voted_nik, tanggal_lahir, jenis_kelamin)

        if validate_nik_structure(voted_nik):
            return voted_nik

    return base_nik


def is_nik_consistent_with_birthdate(
    nik: Optional[str],
    tanggal_lahir_str: Optional[str],
    jenis_kelamin_str: Optional[str] = None
) -> bool:
    """
    Evaluasi bi-directional kelayakan NIK terhadap Tanggal Lahir (DD-MM-YYYY) & Gender.
    Menghasilkan True jika:
    1. NIK 16-digit valid secara struktur (validate_nik_structure).
    2. Tanggal lahir kosong/invalid -> dianggap True (karena tidak ada acuan pembanding).
    3. Tanggal lahir valid -> Bulan (MM) & Tahun (YY) HARUS PERSIS sama antara NIK dan DOB.
       Hari (DD) harus match persis (termasuk +40 untuk perempuan), ATAU jika berbeda,
       perbedaannya HARUS terbatas pada pasangan OCR visual confusion yang valid (5<->6, 1<->7, 3<->8, 0<->8, dll).
    """
    if not nik or len(nik) != 16 or not nik.isdigit():
        return False

    if not validate_nik_structure(nik):
        return False

    if not tanggal_lahir_str or not isinstance(tanggal_lahir_str, str):
        return True

    parts = tanggal_lahir_str.strip().split('-')
    if len(parts) != 3 or len(parts[0]) != 2 or len(parts[1]) != 2 or len(parts[2]) != 4:
        return True

    try:
        d_target = int(parts[0])
        m_target = int(parts[1])
        y_target = parts[2][2:]
    except ValueError:
        return True

    try:
        actual_dd = int(nik[6:8])
        actual_mm = int(nik[8:10])
        actual_yy = nik[10:12]
    except ValueError:
        return False

    # 1. BULAN (MM) & TAHUN (YY) HARUS PERSIS SAMA (Bulan & Tahun tidak boleh beda!)
    if actual_mm != m_target or actual_yy != y_target:
        return False

    # 2. HARI (DD): Target DD Laki-laki (DD) & Perempuan (DD + 40)
    target_dd_male = f"{d_target:02d}"
    target_dd_female = f"{(d_target + 40):02d}"
    actual_dd_str = f"{actual_dd:02d}"

    if jenis_kelamin_str == "PEREMPUAN":
        if actual_dd_str == target_dd_female:
            return True
    elif jenis_kelamin_str == "LAKI-LAKI":
        if actual_dd_str == target_dd_male:
            return True
    else:
        if actual_dd_str in (target_dd_male, target_dd_female):
            return True

    # 3. Toleransi HARI (DD) HANYA untuk Pasangan OCR Visual Confusion
    valid_confusion_pairs = {
        ('5', '6'), ('6', '5'), ('0', '6'), ('6', '0'),
        ('1', '7'), ('7', '1'), ('3', '8'), ('8', '3'),
        ('0', '8'), ('8', '0'), ('1', '9'), ('9', '1'),
        ('0', '5'), ('5', '0'), ('0', '1'), ('1', '0'),
        ('1', '4'), ('4', '1'), ('2', '3'), ('3', '2'),
        ('4', '9'), ('9', '4'), ('5', '9'), ('9', '5'),
        ('0', '9'), ('9', '0'), ('1', '3'), ('3', '1'),
        ('2', '8'), ('8', '2'), ('1', '6'), ('6', '1'),
        ('6', '9'), ('9', '6'),
    }

    targets_to_check = []
    if jenis_kelamin_str == "PEREMPUAN":
        targets_to_check.append(target_dd_female)
    elif jenis_kelamin_str == "LAKI-LAKI":
        targets_to_check.append(target_dd_male)
    else:
        targets_to_check.extend([target_dd_male, target_dd_female])

    for target_dd_str in targets_to_check:
        if len(actual_dd_str) == 2 and len(target_dd_str) == 2:
            diff_indices = [i for i in range(2) if actual_dd_str[i] != target_dd_str[i]]
            if 1 <= len(diff_indices) <= 2:
                if all((actual_dd_str[i], target_dd_str[i]) in valid_confusion_pairs for i in diff_indices):
                    return True

    return False