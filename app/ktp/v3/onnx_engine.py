import io
import threading
import time
import numpy as np
import cv2
from typing import List, Tuple, Optional
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

from app.ktp.v2.paddle_engine import PaddleTextBox


class ONNXEngineV3:
    """
    High-Performance ONNX Runtime OCR Engine for V3 (/ktp/v3/extract).
    Uses PP-OCRv4 detection and recognition models executed via C++ ONNX Runtime (AVX2/AVX-512).
    """
    _instance: Optional['ONNXEngineV3'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_engine()
            return cls._instance

    def _init_engine(self):
        if RapidOCR is None:
            raise RuntimeError("RapidOCR is not installed. Please install rapidocr_onnxruntime.")

        # Initialize RapidOCR with PP-OCRv4 ONNX models
        # - text_score=0.3 matches PaddleOCR det_db_thresh=0.3
        # - use_cls=False avoids unnecessary orientation pass
        self.engine = RapidOCR(text_score=0.3)

    def warmup(self) -> float:
        """Pre-loads ONNX models and warms up C++ runtime graph during container startup."""
        t0 = time.perf_counter()
        dummy_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(dummy_img, "WARMUP KTP ONNX 123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self.engine(dummy_img, use_cls=False)
        return time.perf_counter() - t0

    def extract_text_boxes_with_timing(self, img_bytes: bytes) -> Tuple[List[PaddleTextBox], dict]:
        """
        Executes ONNX Runtime OCR on raw image bytes and returns structured text box list alongside timing metrics.
        Includes smart downscaling to max 960px for sub-second CPU OCR execution.
        """
        t0 = time.perf_counter()

        # Step 1: Decode image bytes to numpy BGR image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        t_decode = time.perf_counter()

        # Step 2: Smart Image Downscaling for OCR Speed Optimization
        max_side = 960
        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)

        scale_factor = 1.0
        if max_dim > max_side:
            scale_factor = max_side / float(max_dim)
            new_w = max(1, int(w * scale_factor))
            new_h = max(1, int(h * scale_factor))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        t_resize = time.perf_counter()

        # Step 3: Execute ONNX Inference (Detection + Recognition)
        results, elapse = self.engine(img_bgr, use_cls=False)
        t_ocr = time.perf_counter()

        text_boxes: List[PaddleTextBox] = []

        # Step 4: Unscale bounding box coordinates back to original image space
        if results:
            for item in results:
                if not item or len(item) < 3:
                    continue
                box, text, score = item[0], item[1], item[2]
                if not text or not str(text).strip():
                    continue

                pts = np.array(box, dtype=np.float32)
                if scale_factor != 1.0:
                    pts = pts / scale_factor

                sc = float(score) * 100.0 if float(score) <= 1.0 else float(score)
                # Strip null bytes and normalize text
                clean_text = str(text).strip().replace("\x00", "")
                text_boxes.append(PaddleTextBox(pts.tolist(), clean_text, sc))

        t_unpack = time.perf_counter()

        timings = {
            "decode_ms": round((t_resize - t0 - (t_resize - t_decode)) * 1000, 2),
            "resize_ms": round((t_resize - t_decode) * 1000, 2),
            "ocr_ms": round((t_ocr - t_resize) * 1000, 2),
            "unpack_ms": round((t_unpack - t_ocr) * 1000, 2),
            "total_engine_ms": round((t_unpack - t0) * 1000, 2),
            "orig_size": f"{w}x{h}",
            "processed_size": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
            "scale": round(scale_factor, 3),
            "det_time_s": round(float(elapse[0]), 3) if elapse and len(elapse) > 0 else 0.0,
            "rec_time_s": round(float(elapse[2]), 3) if elapse and len(elapse) > 2 else 0.0,
        }

        return text_boxes, timings

    def extract_text_boxes(self, img_bytes: bytes) -> List[PaddleTextBox]:
        """Executes ONNX Runtime OCR on raw image bytes and returns structured text box list."""
        boxes, _ = self.extract_text_boxes_with_timing(img_bytes)
        return boxes
