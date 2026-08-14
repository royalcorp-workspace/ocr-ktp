from app.ktp.extractor.common import parse_label_blocks
from app.ktp.extractor.validators import sync_nik_with_birthdate, validate_nik_structure


def test_robust_label_detection():
    raw_text = """
    PROVINSI JAWA BARAT
    KABUPATEN BANDUNG
    N1K: 3204101505950001
    N4MA: BUDI SANTOSO
    T[MPAT/TGL LAHIR: BANDUNG, 15-05-1995
    J3NIS K3LAMIN: LAKI-LAKI
    AGAM4: ISLAM
    """
    blocks = parse_label_blocks(raw_text)
    assert "NIK" in blocks, "NIK label should be detected despite 'N1K'"
    assert "3204101505950001" in blocks["NIK"]
    assert "NAMA" in blocks, "NAMA label should be detected despite 'N4MA'"
    assert "BUDI SANTOSO" in blocks["NAMA"]
    assert "TEMPAT_TGL_LAHIR" in blocks, "TEMPAT_TGL_LAHIR label should be detected despite typo"
    assert "BANDUNG" in blocks["TEMPAT_TGL_LAHIR"]


def test_smart_nik_auto_correction_male():
    # Male: DOB 15-05-1995 -> Target DDMMYY: 150595
    # Corrupted NIK in middle segment: 320410 160595 0001 (16 instead of 15)
    corrupted_nik = "3204101605950001"
    dob_str = "15-05-1995"
    gender_str = "LAKI-LAKI"
    
    corrected = sync_nik_with_birthdate(corrupted_nik, dob_str, gender_str)
    assert corrected == "3204101505950001", f"Expected corrected NIK 3204101505950001, got {corrected}"
    assert validate_nik_structure(corrected)


def test_smart_nik_auto_correction_female():
    # Female: DOB 15-05-1995 -> Target DDMMYY: (15+40)0595 = 550595
    # Corrupted NIK in middle segment: 320410 560595 0001 (56 instead of 55 due to 6/5 confusion)
    corrupted_nik = "3204105605950001"
    dob_str = "15-05-1995"
    gender_str = "PEREMPUAN"
    
    corrected = sync_nik_with_birthdate(corrupted_nik, dob_str, gender_str)
    assert corrected == "3204105505950001", f"Expected corrected NIK 3204105505950001, got {corrected}"
    assert validate_nik_structure(corrected)


def test_smart_nik_auto_correction_invalid_date_digits():
    # NIK with invalid month '13': 320410 151395 0001
    corrupted_nik = "3204101513950001"
    dob_str = "15-05-1995"
    gender_str = "LAKI-LAKI"
    
    corrected = sync_nik_with_birthdate(corrupted_nik, dob_str, gender_str)
    assert corrected == "3204101505950001", f"Expected corrected NIK 3204101505950001, got {corrected}"
    assert validate_nik_structure(corrected)


if __name__ == "__main__":
    test_robust_label_detection()
    test_smart_nik_auto_correction_male()
    test_smart_nik_auto_correction_female()
    test_smart_nik_auto_correction_invalid_date_digits()
    print("All custom unit tests passed successfully!")
