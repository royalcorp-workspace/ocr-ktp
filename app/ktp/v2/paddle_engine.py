import io
import threading
import numpy as np
import cv2
from typing import List, Tuple, Optional
from PIL import Image

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

class PaddleTextBox:
    def __init__(self, box: list, text: str, confidence: float):
        self.box = box
        self.text = text.strip() if text else ""
        self.confidence = float(confidence) if confidence is not None else 0.0

        # Extract coordinates
        pts = np.array(box, dtype=np.float32)
        self.x_min = float(np.min(pts[:, 0]))
        self.x_max = float(np.max(pts[:, 0]))
        self.y_min = float(np.min(pts[:, 1]))
        self.y_max = float(np.max(pts[:, 1]))
        self.center_x = (self.x_min + self.x_max) / 2.0
        self.center_y = (self.y_min + self.y_max) / 2.0
        self.width = max(1.0, self.x_max - self.x_min)
        self.height = max(1.0, self.y_max - self.y_min)

    def __repr__(self):
        return f"<PaddleTextBox text='{self.text}' conf={self.confidence:.2f} y=[{self.y_min:.1f},{self.y_max:.1f}] x=[{self.x_min:.1f},{self.x_max:.1f}]>"

class PaddleEngineV2:
    _instance: Optional['PaddleEngineV2'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_engine()
            return cls._instance

    def _init_engine(self):
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed. Please install paddleocr and paddlepaddle.")
        import os
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "0"
        # Exact runtime match with dockerfile pre-fetch step
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

    def extract_text_boxes(self, img_bytes: bytes) -> List[PaddleTextBox]:
        """
        Executes PaddleOCR on raw image bytes and returns structured text box list.
        """
        # Decode image bytes to numpy BGR image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)
        # RGB to BGR for OpenCV / PaddleOCR compatibility
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = self.ocr.ocr(img_bgr)
        text_boxes: List[PaddleTextBox] = []

        if not results:
            return text_boxes

        first_item = results[0] if isinstance(results, list) and len(results) > 0 else results

        # Format A: PaddleOCR 3.7 dict format {'rec_texts': [...], 'rec_scores': [...], 'dt_polys': [...]}
        if isinstance(first_item, dict) and "rec_texts" in first_item and "rec_scores" in first_item:
            texts = first_item.get("rec_texts", [])
            scores = first_item.get("rec_scores", [])
            polys = first_item.get("dt_polys") if first_item.get("dt_polys") is not None else first_item.get("rec_polys", [])

            for i in range(min(len(texts), len(scores), len(polys))):
                txt = str(texts[i]).strip()
                sc = float(scores[i]) * 100.0 if float(scores[i]) <= 1.0 else float(scores[i])
                box = polys[i]
                if txt:
                    text_boxes.append(PaddleTextBox(box, txt, sc))
            return text_boxes

        # Format B: Legacy 2.x tuple format [[box, (text, conf)], ...]
        items = first_item if isinstance(first_item, list) else [first_item]
        for item in items:
            if not item:
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], (list, tuple)):
                box_coords = item[0]
                text, conf = item[1][0], item[1][1]
                sc = float(conf) * 100.0 if float(conf) <= 1.0 else float(conf)
                if text and str(text).strip():
                    text_boxes.append(PaddleTextBox(box_coords, str(text).strip(), sc))

        return text_boxes
