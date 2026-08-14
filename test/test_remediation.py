import numpy as np
from app.ktp.engine import score_ocr_text
from app.ktp.preprocessing import build_tier1_candidates


def test_remediation_candidate_scoring():
    raw_valid_nik = """
    PROVINSI JAWA BARAT
    KABUPATEN BANDUNG
    NIK: 3204101505950001
    NAMA: BUDI SANTOSO
    TEMPAT/TGL LAHIR: BANDUNG, 15-05-1995
    """
    
    raw_invalid_nik = """
    PROVINSI JAWA BARAT
    KABUPATEN BANDUNG
    NIK: 9999999999999999
    NAMA: BUDI SANTOSO
    TEMPAT/TGL LAHIR: BANDUNG, 15-05-1995
    """
    
    score_valid = score_ocr_text(raw_valid_nik, candidate_name="Pure Grayscale (PSM 6)")
    score_invalid = score_ocr_text(raw_invalid_nik, candidate_name="Pure Grayscale (PSM 6)")
    
    assert score_valid["has_valid_nik"] is True
    assert score_invalid["has_valid_nik"] is False
    assert score_valid["score"] > score_invalid["score"], f"Valid NIK score ({score_valid['score']}) should be strictly greater than invalid NIK score ({score_invalid['score']})"


def test_tier1_candidates_order():
    dummy_img = np.ones((600, 1600, 3), dtype=np.uint8) * 200
    candidates = build_tier1_candidates(dummy_img)
    
    names = [c[0] for c in candidates]
    assert names[0] == "Pure Grayscale (PSM 6)", "Pure Grayscale should be candidate #1"
    assert names[1] == "Blue Channel CLAHE (PSM 6)", "Blue Channel CLAHE should be candidate #2"
    assert names[2] == "V-Channel CLAHE (PSM 6)", "V-Channel CLAHE should be candidate #3"
    assert names[3] == "Morphological Bridged Blue (PSM 6)", "Morphological Bridged Blue should be candidate #4"


if __name__ == "__main__":
    test_remediation_candidate_scoring()
    test_tier1_candidates_order()
    print("All remediation tests passed successfully!")
