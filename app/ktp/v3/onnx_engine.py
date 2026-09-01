import io
import threading
import time
import numpy as np
import cv2
from typing import List, Tuple, Optional
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    import onnxruntime as ort
    from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

    _orig_init_sess_opts = OrtInferSession._init_sess_opts
    def _safe_init_sess_opts(config):
        opt = _orig_init_sess_opts(config)
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        opt.enable_cpu_mem_arena = False
        return opt
    OrtInferSession._init_sess_opts = staticmethod(_safe_init_sess_opts)
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

        # Initialize RapidOCR
        # Check if custom English PP-OCRv4 ONNX model exists in /app/models_onnx/
        import os
        custom_onnx_rec = "/app/models_onnx/en_PP-OCRv4_rec.onnx"
        custom_dict = "/app/models_onnx/en_dict.txt"

        if os.path.exists(custom_onnx_rec):
            kwargs = {"rec_model_path": custom_onnx_rec, "text_score": 0.5}
            if os.path.exists(custom_dict):
                kwargs["rec_keys_path"] = custom_dict
            self.engine = RapidOCR(**kwargs)
        else:
            self.engine = RapidOCR(text_score=0.5)

    @staticmethod
    def _enhance_image(img_bgr: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast enhancement + unsharp masking to improve OCR accuracy
        on low-quality KTP images (poor lighting, worn cards, blur).
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Unsharp masking: sharpen edges without amplifying large noise regions
        blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
        return cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    def warmup(self) -> float:
        """Pre-loads ONNX models and warms up C++ runtime graph during container startup."""
        t0 = time.perf_counter()
        dummy_img = np.ones((128, 320, 3), dtype=np.uint8) * 255
        cv2.putText(dummy_img, "WARMUP KTP ONNX 123", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        self.engine(dummy_img, use_cls=False)
        return time.perf_counter() - t0

    def extract_text_boxes_with_timing(self, img_bytes: bytes) -> Tuple[List[PaddleTextBox], dict]:
        """
        Executes ONNX Runtime OCR on raw image bytes and returns structured text box list alongside timing metrics.
        Includes smart downscaling to max 960px for sub-second CPU OCR execution.
        """
        import unicodedata
        t0 = time.perf_counter()

        # Step 1: Decode image bytes to numpy BGR image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        t_decode = time.perf_counter()

        # Step 2: CLAHE + Unsharp Masking — improves accuracy on low-quality/worn KTP images
        img_bgr = self._enhance_image(img_bgr)

        # Step 3: Proportional downscale to max 960px, then pad to grid-32 (no aspect ratio distortion)
        max_side = 960
        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)

        scale_factor = 1.0
        if max_dim > max_side:
            scale_factor = max_side / float(max_dim)
            new_w = max(1, int(w * scale_factor))
            new_h = max(1, int(h * scale_factor))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w, new_h = w, h

        # Pad to multiple-of-32 using BORDER_REPLICATE to preserve edge context
        pad_w = (32 - new_w % 32) % 32
        pad_h = (32 - new_h % 32) % 32
        if pad_w or pad_h:
            img_bgr = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
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
                # Normalize full-width Unicode characters (e.g. '：' -> ':', 'Ａ' -> 'A')
                normalized_text = unicodedata.normalize('NFKC', str(text)).strip().replace("\x00", "")
                text_boxes.append(PaddleTextBox(pts.tolist(), normalized_text, sc))

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
