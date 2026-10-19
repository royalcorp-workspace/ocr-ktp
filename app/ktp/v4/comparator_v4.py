import re
from typing import Dict, Any, Optional, Tuple
from app.ktp.v4.schemas_v4 import FieldMatchDetailV4, KTPValidationResponseV4


ALIAS_MAPPING: Dict[str, list[str]] = {
    "nik": ["nik", "id_number", "idnumber", "no_ktp", "ktp_number", "identity_number"],
    "nama": ["nama", "name", "full_name", "fullname", "nama_lengkap"],
    "tmp_lahir": ["tmp_lahir", "tempat_lahir", "tmpt_lahir", "birthplace", "birth_place", "pob"],
    "tgl_lahir": ["tgl_lahir", "tanggal_lahir", "birthdate", "birth_date", "dob"],
    "jenis_kelamin": ["jenis_kelamin", "jeniskelamin", "gender", "sex", "kelamin"],
    "agama": ["agama", "religion"],
    "gol_darah": ["gol_darah", "golongan_darah", "goldar", "blood_type", "bloodtype"],
    "alamat": ["alamat", "address", "full_address"],
    "rt_rw": ["rt_rw", "rtrw"],
    "kel_desa": ["kel_desa", "kelurahan_desa", "kelurahan", "desa"],
    "kecamatan": ["kecamatan", "kec"],
    "status_perkawinan": ["status_perkawinan", "status_nikah", "marital_status"],
    "pekerjaan": ["pekerjaan", "occupation", "job"],
    "kewarganegaraan": ["kewarganegaraan", "citizenship", "nationality"]
}

OCR_KEY_MAPPING: Dict[str, str] = {
    "nik": "nik",
    "nama": "nama",
    "tmp_lahir": "tempat_lahir",
    "tgl_lahir": "tanggal_lahir",
    "jenis_kelamin": "jenis_kelamin",
    "agama": "agama",
    "gol_darah": "golongan_darah",
    "alamat": "alamat",
    "rt_rw": "rt_rw",
    "kel_desa": "kelurahan_desa",
    "kecamatan": "kecamatan",
    "status_perkawinan": "status_perkawinan",
    "pekerjaan": "pekerjaan",
    "kewarganegaraan": "kewarganegaraan"
}

CORE_COMPARE_FIELDS = [
    "nik", "nama", "tmp_lahir", "tgl_lahir", "agama", "jenis_kelamin", "gol_darah"
]

FALSY_BLOOD_TYPES = {"-", "--", "---", "", "NONE", "TIDAK TAHU", "TIDAK ADA", "TIDAKTERDETEKSI", "NULL"}


def extract_val(item: Any) -> Optional[str]:
    """Helper to extract clean string value from either string, number, or nested dict."""
    if item is None:
        return None
    if isinstance(item, dict):
        v = item.get("value") if "value" in item else item.get("val")
        return str(v).strip() if v is not None else None
    s = str(item).strip()
    return s if s else None


def normalize_nik(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    digits = re.sub(r'\D', '', val)
    return digits if digits else None


def normalize_name(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.upper().strip()
    # Normalize punctuation commonly preceding titles/suffixes (e.g., "DAMAYANTO, DS")
    s = re.sub(r'[,]', ' ', s)
    s = re.sub(r'[^A-Z\s\.]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def normalize_city(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.upper().strip()
    s = re.sub(r'\b(KOTA|KABUPATEN|KAB\.?)\b', '', s)
    s = re.sub(r'[^A-Z\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def normalize_district(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.upper().strip()
    s = re.sub(r'\b(KECAMATAN|KEC\.?|KELURAHAN|KEL\.?|DESA)\b', '', s)
    s = re.sub(r'[^A-Z0-9\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def normalize_address(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.upper().strip()
    # Harmonize common Indonesian address abbreviations
    s = re.sub(r'\b(JL|JLN)\b\.?', 'JALAN', s)
    s = re.sub(r'\bKP\b\.?', 'KAMPUNG', s)
    s = re.sub(r'\bNOMOR\b', 'NO', s)
    s = re.sub(r'\bNO\b\.?', 'NO', s)
    s = re.sub(r'\bPERUMAHAN\b', 'PERUM', s)
    # Remove dots and commas to prevent punctuation mismatch
    s = re.sub(r'[\.,]', ' ', s)
    # Unify spacing around hyphens (e.g. "A - 9" -> "A-9")
    s = re.sub(r'\s*-\s*', '-', s)
    return re.sub(r'\s+', ' ', s).strip()


def normalize_rt_rw(val: Optional[str]) -> Optional[str]:
    """
    Standard Dukcapil RT/RW normalizer (preserves 3-digit leading zeroes like '000/000').
    Cleans whitespace around the slash character without altering numeric formatting.
    """
    if not val:
        return None
    s = val.upper().strip()
    # Remove RT/RW labels if present (e.g. "RT 001 / RW 007" -> "001/007")
    s = re.sub(r'\b(RT|RW)\b\.?', '', s)
    # Strip whitespace around slash
    s = re.sub(r'\s*/\s*', '/', s)
    # Keep only digits and slash
    s = re.sub(r'[^0-9/]', '', s)
    return s.strip() if s else None


def normalize_date(val: Optional[str]) -> Optional[str]:
    """
    Normalizes date string into canonical 'YYYY-MM-DD' for semantic comparison.
    Supports DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY, etc.
    """
    if not val:
        return None
    s = val.strip()

    # Pattern YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.search(r'(\d{4})[\-\/\.](\d{1,2})[\-\/\.](\d{1,2})', s)
    if m_iso:
        yyyy, mm, dd = m_iso.groups()
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"

    # Pattern DD-MM-YYYY or DD/MM/YYYY
    m_id = re.search(r'(\d{1,2})[\-\/\.](\d{1,2})[\-\/\.](\d{4})', s)
    if m_id:
        dd, mm, yyyy = m_id.groups()
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"

    return s.upper()


def normalize_gender(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    u = val.upper().strip()
    if re.search(r'\b(L|LAKI|PRIA|MALE|M)\b', u):
        return "LAKI-LAKI"
    if re.search(r'\b(P|PEREMPUAN|WANITA|FEMALE|F)\b', u):
        return "PEREMPUAN"
    return u


def normalize_religion(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    u = val.upper().strip()
    if "ISLAM" in u:
        return "ISLAM"
    if "KRISTEN" in u or "PROTESTAN" in u:
        return "KRISTEN"
    if "KATHOLIK" in u or "KATOLIK" in u:
        return "KATHOLIK"
    if "HINDU" in u:
        return "HINDU"
    if "BUDDHA" in u or "BUDHA" in u:
        return "BUDDHA"
    if "KHONGHUCU" in u or "KONGHUCU" in u:
        return "KHONGHUCU"
    return u


def normalize_blood_type(val: Optional[str]) -> Optional[str]:
    if val is None:
        return "-"
    u = val.upper().strip()
    if u in FALSY_BLOOD_TYPES:
        return "-"
    # Token matching
    if "AB" in u:
        return "AB"
    if "A" in u:
        return "A"
    if "B" in u:
        return "B"
    if "O" in u or "0" in u:
        return "O"
    return u


def evaluate_field_equality(field_key: str, ocr_val: Optional[str], db_val: Optional[str]) -> bool:
    """
    Evaluates semantic equality between OCR value and DB/User form value.
    """
    # 1. Golongan Darah Falsy Guard: null, "", "-" matching against each other -> True
    if field_key == "gol_darah":
        norm_ocr_bt = normalize_blood_type(ocr_val)
        norm_db_bt = normalize_blood_type(db_val)
        return norm_ocr_bt == norm_db_bt

    if ocr_val is None and db_val is None:
        return True
    if ocr_val is None or db_val is None:
        return False

    if field_key == "nik":
        n_ocr = normalize_nik(ocr_val)
        n_db = normalize_nik(db_val)
        return bool(n_ocr and n_db and n_ocr == n_db)

    if field_key == "nama":
        n_ocr = normalize_name(ocr_val)
        n_db = normalize_name(db_val)
        if not n_ocr or not n_db:
            return False
        if n_ocr == n_db:
            return True
        # Check initials and suffix equivalence, e.g. "M. RAQIL" vs "M RAQIL", "DAMAYANTO DS." vs "DAMAYANTO DS"
        clean_ocr = re.sub(r'\.', '', n_ocr).strip()
        clean_db = re.sub(r'\.', '', n_db).strip()
        if clean_ocr == clean_db:
            return True
        return clean_ocr.split() == clean_db.split()

    if field_key == "tmp_lahir":
        c_ocr = normalize_city(ocr_val)
        c_db = normalize_city(db_val)
        if not c_ocr or not c_db:
            return False
        return c_ocr == c_db or c_ocr in c_db or c_db in c_ocr

    if field_key == "tgl_lahir":
        d_ocr = normalize_date(ocr_val)
        d_db = normalize_date(db_val)
        return bool(d_ocr and d_db and d_ocr == d_db)

    if field_key == "jenis_kelamin":
        g_ocr = normalize_gender(ocr_val)
        g_db = normalize_gender(db_val)
        return bool(g_ocr and g_db and g_ocr == g_db)

    if field_key == "agama":
        r_ocr = normalize_religion(ocr_val)
        r_db = normalize_religion(db_val)
        return bool(r_ocr and r_db and r_ocr == r_db)

    if field_key == "alamat":
        a_ocr = normalize_address(ocr_val)
        a_db = normalize_address(db_val)
        if not a_ocr or not a_db:
            return False
        if a_ocr == a_db:
            return True
        # Tolerant of hyphen vs space (e.g. BLOK A-9 vs BLOK A 9)
        return a_ocr.replace('-', ' ') == a_db.replace('-', ' ')

    if field_key == "rt_rw":
        rw_ocr = normalize_rt_rw(ocr_val)
        rw_db = normalize_rt_rw(db_val)
        if not rw_ocr or not rw_db:
            return False
        return rw_ocr == rw_db

    if field_key in ("kecamatan", "kel_desa"):
        kd_ocr = normalize_district(ocr_val)
        kd_db = normalize_district(db_val)
        if not kd_ocr or not kd_db:
            return False
        return kd_ocr == kd_db or kd_ocr in kd_db or kd_db in kd_ocr

    # Generic fallback
    return str(ocr_val).strip().upper() == str(db_val).strip().upper()


def resolve_db_field_value(field_key: str, db_input: Dict[str, Any]) -> Optional[str]:
    """Looks up value from db_input using primary key and aliases."""
    aliases = ALIAS_MAPPING.get(field_key, [field_key])
    for alias in aliases:
        if alias in db_input:
            val = extract_val(db_input[alias])
            if val is not None:
                return val
        # Check case-insensitive
        for k, v in db_input.items():
            if k.lower() == alias.lower():
                val = extract_val(v)
                if val is not None:
                    return val
    return None


def is_field_in_db_input(field_key: str, db_input: Dict[str, Any]) -> bool:
    """Checks if field or any of its aliases was explicitly provided in db_input."""
    aliases = ALIAS_MAPPING.get(field_key, [field_key])
    for alias in aliases:
        if alias in db_input:
            return True
        for k in db_input.keys():
            if k.lower() == alias.lower():
                return True
    return False


def compare_ktp_fields(ocr_dict: Dict[str, Any], db_input: Dict[str, Any]) -> KTPValidationResponseV4:
    """
    Compares OCR extraction against DB input across target Dukcapil fields.
    Returns KTPValidationResponseV4 contract for frontend pop-up validation.
    """
    details: Dict[str, FieldMatchDetailV4] = {}
    mismatched_fields: list[str] = []

    # Determine which fields to evaluate: Core fields + any additional Dukcapil fields present in db_input
    fields_to_evaluate = list(CORE_COMPARE_FIELDS)
    for extra_field in ["alamat", "rt_rw", "kel_desa", "kecamatan", "status_perkawinan", "pekerjaan", "kewarganegaraan"]:
        if is_field_in_db_input(extra_field, db_input):
            if extra_field not in fields_to_evaluate:
                fields_to_evaluate.append(extra_field)

    for field_key in fields_to_evaluate:
        # If field was not provided in db_input, do not evaluate
        if not is_field_in_db_input(field_key, db_input):
            continue

        raw_db = resolve_db_field_value(field_key, db_input)
        ocr_field_name = OCR_KEY_MAPPING.get(field_key, field_key)
        ocr_item = ocr_dict.get(ocr_field_name)
        raw_ocr = extract_val(ocr_item)

        is_match = evaluate_field_equality(field_key, raw_ocr, raw_db)

        if is_match:
            details[field_key] = FieldMatchDetailV4(match=True)
        else:
            details[field_key] = FieldMatchDetailV4(
                match=False,
                ocr=raw_ocr,
                db=raw_db
            )
            mismatched_fields.append(field_key)

    is_different = len(mismatched_fields) > 0

    return KTPValidationResponseV4(
        different=is_different,
        mismatched_fields=mismatched_fields,
        details=details
    )
