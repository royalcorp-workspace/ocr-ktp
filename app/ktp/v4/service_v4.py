from typing import Dict, Any
from app.ktp.v3.service_v3 import process_ktp_image_v3
from app.ktp.v4.comparator_v4 import compare_ktp_fields
from app.ktp.v4.schemas_v4 import KTPValidationResponseV4


def run_ktp_validation_v4(img_bytes: bytes, db_input: Dict[str, Any]) -> KTPValidationResponseV4:
    """
    Executes V4 KTP Validation:
    1. Runs stabilized V3 OCR engine extraction on the KTP image bytes.
    2. Compares extracted OCR fields against input DB/user form data.
    3. Produces specialized frontend validation schema (different, mismatched_fields, details).
    """
    # 1. OCR Extraction using V3
    ocr_response = process_ktp_image_v3(img_bytes)
    ocr_dict = ocr_response.model_dump()

    # 2. Multi-field comparison
    validation_result = compare_ktp_fields(ocr_dict, db_input)
    return validation_result
