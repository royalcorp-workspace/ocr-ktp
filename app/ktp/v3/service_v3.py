import time
from typing import Dict, Any
from app.core.logging_config import logger, request_id_var
from app.ktp.v3.onnx_engine import ONNXEngineV3
from app.ktp.v3.spatial_parser_v3 import SpatialParserV3
from app.ktp.v2.schemas_v2 import KTPOcrResponseV2, ConsensusResponseV2, FieldWithSourceV2
from app.ktp.v2.consensus_v2 import run_consensus_v2

FIELD_NAMES = [
    "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
    "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
    "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
]

CRITICAL_FIELDS = [
    "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
    "alamat", "rt_rw", "kelurahan_desa", "kecamatan", "agama",
    "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
]


def _run_tiered_hybrid_pipeline(img_bytes: bytes) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Tiered Hybrid Pipeline:
    - Tier 1: V3 ONNX Engine (0.3s)
    - Tier 2: V2 Paddle Engine Fallback (triggered only if 2+ critical fields are missing)
    """
    req_id = request_id_var.get()
    t_start = time.perf_counter()

    # Tier 1: V3 Fast-Path
    engine_v3 = ONNXEngineV3()
    text_boxes, timings = engine_v3.extract_text_boxes_with_timing(img_bytes)

    t_parse_start = time.perf_counter()
    parser_v3 = SpatialParserV3()
    parsed_raw = parser_v3.parse_ktp(text_boxes)
    t_parse_ms = round((time.perf_counter() - t_parse_start) * 1000, 2)

    missing_fields = [fn for fn in CRITICAL_FIELDS if not parsed_raw.get(fn, {}).get("val")]
    is_hybrid_triggered = False

    # Tier 2: Deep Fallback if 2 or more critical fields are missing
    if len(missing_fields) >= 2:
        try:
            from app.ktp.v2.paddle_engine import PaddleEngineV2
            from app.ktp.v2.spatial_parser import SpatialParserV2

            logger.info(f"[Hybrid Fallback] Tier 2 triggered for req_id={req_id}, missing={missing_fields}")
            p_engine = PaddleEngineV2()
            v2_boxes = p_engine.extract_text_boxes(img_bytes)
            v2_parser = SpatialParserV2()
            v2_raw = v2_parser.parse_ktp(v2_boxes)

            # Merge: V2 patches missing fields or significantly higher confidence fields
            for fn in FIELD_NAMES:
                v3_item = parsed_raw.get(fn, {"val": None, "conf": 0.0})
                v2_item = v2_raw.get(fn, {"val": None, "conf": 0.0})

                if not v3_item.get("val") and v2_item.get("val"):
                    parsed_raw[fn] = v2_item
                elif v3_item.get("val") and v2_item.get("val"):
                    if float(v2_item.get("conf", 0.0)) > float(v3_item.get("conf", 0.0)) + 15.0:
                        parsed_raw[fn] = v2_item
            is_hybrid_triggered = True
        except Exception as e:
            logger.warning(f"[Hybrid Fallback] Tier 2 fallback skipped/unavailable: {e}")

    t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    timings_summary = {
        "req_id": req_id,
        "total_ms": t_total_ms,
        "spatial_parse_ms": t_parse_ms,
        "detected_boxes": len(text_boxes),
        "hybrid_fallback": is_hybrid_triggered,
        **timings
    }
    return parsed_raw, timings_summary


from typing import Tuple


def process_ktp_image_v3(img_bytes: bytes) -> KTPOcrResponseV2:
    """
    Extracts 15 Dukcapil fields using Tiered Hybrid Engine.
    """
    parsed_raw, timings = _run_tiered_hybrid_pipeline(img_bytes)
    req_id = timings.get("req_id")

    logger.info(
        f"[OCR Extract Breakdown] req_id={req_id} total_ms={timings.get('total_ms')} "
        f"hybrid={timings.get('hybrid_fallback')} boxes={timings.get('detected_boxes')} "
        f"ocr_ms={timings.get('ocr_ms')}"
    )

    response_dict = {}
    for fn in FIELD_NAMES:
        item = parsed_raw.get(fn, {"val": None, "conf": 0.0})
        response_dict[fn] = FieldWithSourceV2(
            value=item["val"],
            confidence=float(item["conf"]),
            source="OCR"
        )

    return KTPOcrResponseV2(**response_dict)


def run_consensus_ocr_v3(img_bytes: bytes, mobile_data: dict) -> ConsensusResponseV2:
    """
    Consensus Validator using Tiered Hybrid Engine.
    """
    parsed_raw, timings = _run_tiered_hybrid_pipeline(img_bytes)
    req_id = timings.get("req_id")

    t_cons_start = time.perf_counter()
    consensus_payload = run_consensus_v2(parsed_raw, mobile_data or {})
    t_cons_ms = round((time.perf_counter() - t_cons_start) * 1000, 2)

    logger.info(
        f"[OCR Validate Breakdown] req_id={req_id} total_ms={timings.get('total_ms')} "
        f"consensus_ms={t_cons_ms} hybrid={timings.get('hybrid_fallback')}"
    )

    return ConsensusResponseV2(success=True, data=consensus_payload)
