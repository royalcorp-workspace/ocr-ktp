from app.api.ktp_routes import _calculate_field_confidence


def test_dynamic_confidence_calculation():
    # Sample 1: High OCR word confidence from Tesseract (e.g. 96.5%) with matching DOB
    word_conf_map_1 = {
        "3204101505950001": 96.5,
        "BUDI": 94.0,
        "SANTOSO": 92.5,
        "BANDUNG": 91.0,
        "15-05-1995": 95.0,
    }
    all_fields_1 = {
        "nik": "3204101505950001",
        "nama": "BUDI SANTOSO",
        "tempat_lahir": "BANDUNG",
        "tanggal_lahir": "15-05-1995",
    }
    
    conf_nik_1 = _calculate_field_confidence("nik", "3204101505950001", base_score=70, word_conf_map=word_conf_map_1, all_fields=all_fields_1)
    conf_nama_1 = _calculate_field_confidence("nama", "BUDI SANTOSO", base_score=70, word_conf_map=word_conf_map_1, all_fields=all_fields_1)
    conf_tempat_1 = _calculate_field_confidence("tempat_lahir", "BANDUNG", base_score=70, word_conf_map=word_conf_map_1, all_fields=all_fields_1)
    conf_tanggal_1 = _calculate_field_confidence("tanggal_lahir", "15-05-1995", base_score=70, word_conf_map=word_conf_map_1, all_fields=all_fields_1)
    
    # Sample 2: Lower OCR word confidence from Tesseract (e.g. 78.0%)
    word_conf_map_2 = {
        "3204101505950001": 78.0,
        "BUDI": 72.0,
        "SANTOSO": 70.5,
        "BANDUNG": 68.0,
        "15-05-1995": 81.0,
    }
    all_fields_2 = {
        "nik": "3204101505950001",
        "nama": "BUDI SANTOSO",
        "tempat_lahir": "BANDUNG",
        "tanggal_lahir": "15-05-1995",
    }
    
    conf_nik_2 = _calculate_field_confidence("nik", "3204101505950001", base_score=50, word_conf_map=word_conf_map_2, all_fields=all_fields_2)
    conf_nama_2 = _calculate_field_confidence("nama", "BUDI SANTOSO", base_score=50, word_conf_map=word_conf_map_2, all_fields=all_fields_2)
    conf_tempat_2 = _calculate_field_confidence("tempat_lahir", "BANDUNG", base_score=50, word_conf_map=word_conf_map_2, all_fields=all_fields_2)
    conf_tanggal_2 = _calculate_field_confidence("tanggal_lahir", "15-05-1995", base_score=50, word_conf_map=word_conf_map_2, all_fields=all_fields_2)

    # Verify values are dynamic and NOT identical static numbers
    assert conf_nik_1 != conf_nik_2, f"NIK confidence should be dynamic! ({conf_nik_1} vs {conf_nik_2})"
    assert conf_nama_1 != conf_nama_2, f"Nama confidence should be dynamic! ({conf_nama_1} vs {conf_nama_2})"
    assert conf_tempat_1 != conf_tempat_2, f"Tempat Lahir confidence should be dynamic! ({conf_tempat_1} vs {conf_tempat_2})"
    assert conf_tanggal_1 != conf_tanggal_2, f"Tanggal Lahir confidence should be dynamic! ({conf_tanggal_1} vs {conf_tanggal_2})"
    
    assert conf_nik_1 > conf_nik_2, "Sample 1 should have higher confidence than Sample 2"
    assert conf_nik_1 >= 85.0, f"Perfect sample NIK should have high confidence: {conf_nik_1}"


def test_edit_distance_mutation_penalty():
    # Scenario: Raw OCR token in word_conf_map was "3204101505940001" (last digit '4')
    # Corrected value is "3204101505950001" (last digit '5') -> Edit distance = 1
    word_conf_map_perfect = {"3204101505950001": 95.0}
    word_conf_map_mutated = {"3204101505940001": 95.0}

    conf_perfect = _calculate_field_confidence("nik", "3204101505950001", base_score=70, word_conf_map=word_conf_map_perfect)
    conf_mutated = _calculate_field_confidence("nik", "3204101505950001", base_score=70, word_conf_map=word_conf_map_mutated)

    assert conf_mutated < conf_perfect, f"Mutated NIK should get lower confidence! ({conf_mutated} vs {conf_perfect})"
    # Mutation penalty for edit distance 1 is 12.0
    assert round(conf_perfect - conf_mutated, 1) >= 10.0, f"Mutation penalty should reduce score significantly ({conf_perfect} -> {conf_mutated})"


def test_character_anomaly_penalty():
    # Raw OCR text contains alpha noise in NIK: "32O41O15O5950001" (3 'O's instead of '0's)
    raw_text_noisy = "PROVINSI JAWA BARAT\nNIK 32O41O15O5950001\nNAMA BUDI SANTOSO"
    raw_text_clean = "PROVINSI JAWA BARAT\nNIK 3204101505950001\nNAMA BUDI SANTOSO"

    conf_noisy = _calculate_field_confidence("nik", "3204101505950001", base_score=70, raw_text=raw_text_noisy)
    conf_clean = _calculate_field_confidence("nik", "3204101505950001", base_score=70, raw_text=raw_text_clean)

    assert conf_noisy < conf_clean, f"Noisy raw NIK should get lower confidence! ({conf_noisy} vs {conf_clean})"


def test_cross_validation_dob_mismatch_penalty():
    # NIK DOB is 15-05-1995 (digit 7-12: 150595)
    # Field DOB is 20-10-1990 (mismatch!)
    all_fields_match = {
        "nik": "3204101505950001",
        "tanggal_lahir": "15-05-1995",
    }
    all_fields_mismatch = {
        "nik": "3204101505950001",
        "tanggal_lahir": "20-10-1990",
    }

    conf_match = _calculate_field_confidence("nik", "3204101505950001", base_score=70, all_fields=all_fields_match)
    conf_mismatch = _calculate_field_confidence("nik", "3204101505950001", base_score=70, all_fields=all_fields_mismatch)

    assert conf_mismatch <= 55.0, f"Cross-validation mismatch should cap confidence at <= 55.0%! Got: {conf_mismatch}"
    assert conf_mismatch < conf_match, f"Match should be higher than mismatch! ({conf_match} vs {conf_mismatch})"


def test_invalid_nik_structure_gated_cap():
    # NIK length 15 digits (invalid structure)
    conf_invalid_len = _calculate_field_confidence("nik", "320410150595000", base_score=80)
    assert conf_invalid_len <= 45.0, f"Invalid NIK length should cap confidence at <= 45.0%! Got: {conf_invalid_len}"


if __name__ == "__main__":
    test_dynamic_confidence_calculation()
    test_edit_distance_mutation_penalty()
    test_character_anomaly_penalty()
    test_cross_validation_dob_mismatch_penalty()
    test_invalid_nik_structure_gated_cap()
    print("All dynamic confidence unit tests passed successfully!")

