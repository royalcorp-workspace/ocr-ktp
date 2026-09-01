import time
from typing import Dict, Any
from app.core.logging_config import logger, request_id_var
from app.ktp.v2.paddle_engine import PaddleEngineV2
from app.ktp.v2.spatial_parser import SpatialParserV2
from app.ktp.v2.schemas_v2 import KTPOcrResponseV2, ConsensusResponseV2, FieldWithSourceV2
from app.ktp.v2.consensus_v2 import run_consensus_v2

def process_ktp_image_v2(img_bytes: bytes) -> KTPOcrResponseV2:
    """
    Core OCR Extraction pipeline for V2 (/ktp/v2/extract).
    Always returns source="OCR" for all fields.
    """
    req_id = request_id_var.get()
    t_start = time.perf_counter()

    engine = PaddleEngineV2()
    text_boxes, timings = engine.extract_text_boxes_with_timing(img_bytes)

    t_parse_start = time.perf_counter()
    parser = SpatialParserV2()
    parsed_raw = parser.parse_ktp(text_boxes)
    t_parse_ms = round((time.perf_counter() - t_parse_start) * 1000, 2)

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

    t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        f"[V2 Extract Breakdown] req_id={req_id} total_ms={t_total_ms} "
        f"(orig_size={timings.get('orig_size')} processed_size={timings.get('processed_size')} "
        f"scale={timings.get('scale')} decode_ms={timings.get('decode_ms')} resize_ms={timings.get('resize_ms')} "
        f"ocr_infer_ms={timings.get('ocr_ms')} unpack_boxes_ms={timings.get('unpack_ms')} "
        f"spatial_parse_ms={t_parse_ms} detected_boxes={len(text_boxes)})"
    )

    return KTPOcrResponseV2(**response_dict)

def run_consensus_ocr_v2(img_bytes: bytes, mobile_data: dict) -> ConsensusResponseV2:
    """
    Consensus Validator pipeline for V2 (/ktp/v2/validate).
    Returns source="OCR" or source="MOBILE".
    """
    req_id = request_id_var.get()
    t_start = time.perf_counter()

    engine = PaddleEngineV2()
    text_boxes, timings = engine.extract_text_boxes_with_timing(img_bytes)

    t_parse_start = time.perf_counter()
    parser = SpatialParserV2()
    parsed_raw = parser.parse_ktp(text_boxes)
    t_parse_ms = round((time.perf_counter() - t_parse_start) * 1000, 2)

    t_cons_start = time.perf_counter()
    consensus_payload = run_consensus_v2(parsed_raw, mobile_data or {})
    t_cons_ms = round((time.perf_counter() - t_cons_start) * 1000, 2)

    t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        f"[V2 Validate Breakdown] req_id={req_id} total_ms={t_total_ms} "
        f"(orig_size={timings.get('orig_size')} processed_size={timings.get('processed_size')} "
        f"ocr_infer_ms={timings.get('ocr_ms')} spatial_parse_ms={t_parse_ms} consensus_ms={t_cons_ms})"
    )

    return ConsensusResponseV2(success=True, data=consensus_payload)
