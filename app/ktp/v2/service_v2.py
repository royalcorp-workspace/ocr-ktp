from typing import Dict, Any
from app.ktp.v2.paddle_engine import PaddleEngineV2
from app.ktp.v2.spatial_parser import SpatialParserV2
from app.ktp.v2.schemas_v2 import KTPOcrResponseV2, ConsensusResponseV2, FieldWithSourceV2
from app.ktp.v2.consensus_v2 import run_consensus_v2

def process_ktp_image_v2(img_bytes: bytes) -> KTPOcrResponseV2:
    """
    Core OCR Extraction pipeline for V2 (/ktp/v2/extract).
    Always returns source="OCR" for all fields.
    """
    engine = PaddleEngineV2()
    text_boxes = engine.extract_text_boxes(img_bytes)

    parser = SpatialParserV2()
    parsed_raw = parser.parse_ktp(text_boxes)

    field_names = [
        "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
        "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
        "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
    ]

    response_dict = {}
    for fn in field_names:
        item = parsed_raw.get(fn, {"val": None, "conf": 0.0})
        response_dict[fn] = FieldWithSourceV2(
            value=item["val"],
            confidence=float(item["conf"]),
            source="OCR"
        )

    return KTPOcrResponseV2(**response_dict)

def run_consensus_ocr_v2(img_bytes: bytes, mobile_data: dict) -> ConsensusResponseV2:
    """
    Consensus Validator pipeline for V2 (/ktp/v2/validate).
    Returns source="OCR" or source="MOBILE".
    """
    engine = PaddleEngineV2()
    text_boxes = engine.extract_text_boxes(img_bytes)

    parser = SpatialParserV2()
    parsed_raw = parser.parse_ktp(text_boxes)

    consensus_payload = run_consensus_v2(parsed_raw, mobile_data or {})
    return ConsensusResponseV2(success=True, data=consensus_payload)
