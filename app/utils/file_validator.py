import io
import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException, status
from app.core.logging_config import logger

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

def is_valid_image_header(content: bytes) -> bool:
    """
    Verifikasi cepat berbasis Magic Bytes (Header Biner Gambar).
    Mendukung format: JPEG, PNG, WEBP, BMP.
    """
    if len(content) < 4:
        return False
    
    # JPEG: \xff\xd8\xff
    if content.startswith(b"\xff\xd8\xff"):
        return True
    
    # PNG: \x89PNG
    if content.startswith(b"\x89PNG"):
        return True
    
    # BMP: BM
    if content.startswith(b"BM"):
        return True
    
    # WEBP: RIFF....WEBP
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return True
        
    return False


def validate_image_bytes(content: bytes, max_file_size: int = MAX_FILE_SIZE_BYTES) -> np.ndarray:
    """
    Validasi binary image content secara mendalam:
    1. Cek keberadaan & ukuran file (0 bytes -> 400, > 10MB -> 413).
    2. Header Magic Bytes check.
    3. Deep Decoding via OpenCV / PIL.
    
    Mengembalikan numpy array (BGR Image) jika gambar valid.
    Melempar HTTPException (400, 413, 415) jika invalid.
    """
    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File gambar kosong atau tidak dapat dibaca."
        )

    if len(content) > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Ukuran file terlalu besar, maksimal {max_file_size // (1024 * 1024)} MB."
        )

    # 1. Header Check (Magic Bytes)
    has_valid_header = is_valid_image_header(content)

    # 2. Deep Decoding Check via OpenCV
    img = None
    try:
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"cv2.imdecode decoding gagal: {e}")

    # Fallback Check via PIL if cv2 returned None
    if img is None:
        try:
            pil_img = Image.open(io.BytesIO(content))
            pil_img.verify()
            pil_img = Image.open(io.BytesIO(content))
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"PIL Image verification gagal: {e}")

    # Jika dekoding gagal atau tidak valid sebagai matriks gambar
    if img is None or img.size == 0 or not hasattr(img, "shape"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Tipe file tidak didukung. Harap unggah gambar JPEG, PNG, WebP, atau BMP yang valid."
        )

    return img
