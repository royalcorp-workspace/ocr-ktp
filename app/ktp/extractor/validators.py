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


def score_nik_candidate(nik: str) -> float:
    """
    Hitung skor validasi internal NIK berbasis UU Adminduk / Permendagri.
    SKOR MAKSIMAL: 100.0.
    1. 16-Digit Purity (+30)
    2. Kode Wilayah Valid PPKKCC (+20)
    3. Tanggal Lahir (Pria 01-31, Wanita 41-71) (+15)
    4. Bulan Lahir (01-12) (+15)
    5. Validitas Kalender datetime.date (+10)
    6. Nomor Urut NNNN != 0000 (+10)
    """
    if not nik or not str(nik).isdigit():
        return 0.0

    s = str(nik).strip()
    if len(s) != 16:
        return 10.0 if len(s) in [15, 17] else 0.0

    score = 30.0  # 16-digit numeric base score

    try:
        prov = int(s[0:2])
        kab = int(s[2:4])
        kec = int(s[4:6])
        dd = int(s[6:8])
        mm = int(s[8:10])
        yy = int(s[10:12])
        seq = int(s[12:16])

        # Kode Wilayah (PPKKCC)
        if (11 <= prov <= 92) and (1 <= kab <= 78) and (1 <= kec <= 75):
            score += 20.0

        # Tanggal Lahir (Pria 01-31, Wanita 41-71)
        real_dd = dd - 40 if dd > 40 else dd
        if (1 <= dd <= 31) or (41 <= dd <= 71):
            score += 15.0

        # Bulan Lahir (01-12)
        if 1 <= mm <= 12:
            score += 15.0

        # Validitas Kalender Legal (datetime.date)
        import datetime
        yyyy = yy + 1900 if yy > 26 else yy + 2000
        try:
            datetime.date(yyyy, mm, real_dd)
            score += 10.0
        except ValueError:
            score -= 20.0

        # Nomor Urut NNNN != 0000
        if 1 <= seq <= 9999:
            score += 10.0
        else:
            score -= 20.0

    except ValueError:
        return 0.0

    return max(0.0, min(100.0, score))


def check_nik_dob_consistency(
    nik: Optional[str],
    tanggal_lahir_str: Optional[str],
    jenis_kelamin_str: Optional[str] = None
) -> bool:
    """
    Evaluasi konsistensi read-only antara NIK (DDMMYY) dan Tanggal Lahir (DD-MM-YYYY).
    TIDAK MENGUBAH / MUTASI digit NIK.
    """
    if not nik or len(nik) != 16 or not nik.isdigit():
        return False
    if not tanggal_lahir_str:
        return False

    parts = tanggal_lahir_str.split('-')
    if len(parts) != 3 or len(parts[0]) != 2 or len(parts[1]) != 2 or len(parts[2]) != 4:
        return False

    try:
        d_target = int(parts[0])
        m_target = int(parts[1])
        y_target = parts[2][2:]
    except ValueError:
        return False

    try:
        actual_dd = int(nik[6:8])
        actual_mm = int(nik[8:10])
        actual_yy = nik[10:12]
    except ValueError:
        return False

    real_dd = actual_dd - 40 if actual_dd > 40 else actual_dd
    if real_dd == d_target and actual_mm == m_target and actual_yy == y_target:
        return True

    return False


def sync_nik_with_birthdate(
    nik: Optional[str],
    tanggal_lahir_str: Optional[str],
    jenis_kelamin_str: Optional[str]
) -> Optional[str]:
    """
    DEPRECATED MUTATION FUNCTION: Read-only passthrough for backward compatibility.
    AKAN SELALU mengembalikan NIK asli dari OCR tanpa melakukan mutasi digit.
    """
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
    
    if tanggal_lahir and is_nik_consistent_with_birthdate(base_nik, tanggal_lahir, jenis_kelamin):
        return base_nik

    if len(nik_candidates) >= 1:
        voted_chars = list(base_nik)
        for idx in range(16):
            if 6 <= idx <= 11 and tanggal_lahir and is_nik_consistent_with_birthdate(base_nik, tanggal_lahir, jenis_kelamin):
                continue
            char_weights = {}
            for rank, n_str in enumerate(nik_candidates):
                if len(n_str) == 16:
                    c = n_str[idx]
                    weight = 1.0 - (rank * 0.01)
                    char_weights[c] = char_weights.get(c, 0.0) + weight
            
            if char_weights:
                voted_chars[idx] = max(char_weights.keys(), key=lambda k: char_weights[k])
            
        voted_nik = "".join(voted_chars)

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