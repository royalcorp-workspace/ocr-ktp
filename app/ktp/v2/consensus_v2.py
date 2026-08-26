from typing import Dict, Any, Optional, Tuple
from app.ktp.v2.schemas_v2 import KTPOcrResponseV2, FieldWithSourceV2
from app.ktp.v2.field_cleaners import clean_nik

def calculate_fuzzy_similarity(str1: str, str2: str) -> float:
    """Calculates Levenshtein similarity ratio between 0.0 and 1.0."""
    if not str1 or not str2:
        return 0.0
    s1, s2 = str1.upper().strip(), str2.upper().strip()
    if s1 == s2:
        return 1.0
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    distance = dp[m][n]
    max_len = max(m, n)
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - (distance / max_len))

def extract_val_from_input(item: Any) -> Optional[str]:
    """Helper to extract clean string value from either string or nested dict input."""
    if item is None:
        return None
    if isinstance(item, dict):
        v = item.get("value") if "value" in item else item.get("val")
        return str(v).strip() if v is not None else None
    s = str(item).strip()
    return s if s else None

def evaluate_nik_consensus(
    ocr_val: Optional[str],
    ocr_conf: float,
    mobile_val: Optional[str],
    dob_val: Any = None
) -> Tuple[Optional[str], float, str]:
    """
    Evaluates NIK consensus using 4-state NIK validity matrix (Section 7).
    """
    str_ocr = extract_val_from_input(ocr_val)
    str_mob = extract_val_from_input(mobile_val)

    c_ocr = clean_nik(str_ocr)
    c_mob = clean_nik(str_mob)

    ocr_valid = bool(c_ocr and len(c_ocr) == 16)
    mob_valid = bool(c_mob and len(c_mob) == 16)

    # 1. State: Tidak - Tidak (Both Invalid 16d)
    if not ocr_valid and not mob_valid:
        if c_ocr:
            return c_ocr, min(ocr_conf, 50.0), "OCR"
        elif c_mob:
            return c_mob, 100.0, "MOBILE"
        else:
            return None, 0.0, "OCR"

    # 2. State: Tidak - Ya (Mobile Valid, OCR Invalid)
    if not ocr_valid and mob_valid:
        return c_mob, 100.0, "MOBILE"

    # 3. State: Ya - Tidak (OCR Valid, Mobile Invalid)
    if ocr_valid and not mob_valid:
        return c_ocr, ocr_conf, "OCR"

    # 4. State: Ya - Ya (Both Valid 16d)
    if c_ocr == c_mob:
        return c_ocr, ocr_conf, "OCR"

    # Both valid 16d but different digits
    # Check DOB match if available
    str_dob = extract_val_from_input(dob_val)
    if str_dob and len(str_dob.split("-")) == 3:
        try:
            d, m, y = str_dob.split("-")
            dob_nik_part = f"{d}{m}{y[2:]}"
            ocr_dob_match = dob_nik_part in c_ocr
            mob_dob_match = dob_nik_part in c_mob

            if ocr_dob_match and not mob_dob_match:
                return c_ocr, ocr_conf, "OCR"
            elif mob_dob_match and not ocr_dob_match:
                return c_mob, 100.0, "MOBILE"
        except Exception:
            pass

    if ocr_conf >= 85.0:
        return c_ocr, ocr_conf, "OCR"
    else:
        return c_mob, 100.0, "MOBILE"

def evaluate_field_consensus(
    field_name: str,
    ocr_data: Dict[str, Any],
    mobile_data: Dict[str, Any],
    dob_val: Any = None
) -> FieldWithSourceV2:
    """
    Evaluates field consensus using State Matrix (Section 5) and Truth Table (Section 6).
    """
    ocr_val = extract_val_from_input(ocr_data.get("val") if "val" in ocr_data else ocr_data.get("value"))
    ocr_conf = float(ocr_data.get("conf", 0.0)) if "conf" in ocr_data else float(ocr_data.get("confidence", 0.0))
    mobile_val = extract_val_from_input(mobile_data.get(field_name))

    # State 1: OCR Null + Mobile Null
    if not ocr_val and not mobile_val:
        return FieldWithSourceV2(value=None, confidence=0.0, source="OCR")

    # State 2: OCR Null + Mobile Has Value
    if not ocr_val and mobile_val:
        return FieldWithSourceV2(value=mobile_val, confidence=100.0, source="MOBILE")

    # State 3: OCR Has Value + Mobile Null
    if ocr_val and not mobile_val:
        return FieldWithSourceV2(value=ocr_val, confidence=ocr_conf, source="OCR")

    # State 4: Both OCR and Mobile Have Values
    if field_name == "nik":
        v, c, s = evaluate_nik_consensus(ocr_val, ocr_conf, mobile_val, dob_val)
        return FieldWithSourceV2(value=v, confidence=c, source=s)

    # Exact Match
    if ocr_val.upper() == mobile_val.upper():
        return FieldWithSourceV2(value=ocr_val, confidence=ocr_conf, source="OCR")

    # Text Field Mismatch: Evaluate Truth Table (Section 6)
    sim = calculate_fuzzy_similarity(ocr_val, mobile_val)

    # High Similarity (>= 80%)
    if sim >= 0.80:
        return FieldWithSourceV2(value=ocr_val, confidence=ocr_conf, source="OCR")

    # Borderline Similarity (60% <= sim < 80%)
    if 0.60 <= sim < 0.80:
        if ocr_conf >= 85.0:
            return FieldWithSourceV2(value=ocr_val, confidence=ocr_conf, source="OCR")
        else:
            return FieldWithSourceV2(value=mobile_val, confidence=100.0, source="MOBILE")

    # Divergent (< 60%)
    return FieldWithSourceV2(value=mobile_val, confidence=100.0, source="MOBILE")

def run_consensus_v2(ocr_payload: Dict[str, Dict[str, Any]], mobile_input: Dict[str, Any]) -> KTPOcrResponseV2:
    """
    Executes Consensus V2 over all 15 fields.
    """
    field_names = [
        "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
        "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
        "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
    ]

    response_data = {}
    dob_val = extract_val_from_input(ocr_payload.get("tanggal_lahir", {}).get("val")) or extract_val_from_input(mobile_input.get("tanggal_lahir"))

    for fn in field_names:
        ocr_f = ocr_payload.get(fn, {"val": None, "conf": 0.0})
        res_field = evaluate_field_consensus(fn, ocr_f, mobile_input, dob_val)
        response_data[fn] = res_field

    return KTPOcrResponseV2(**response_data)
