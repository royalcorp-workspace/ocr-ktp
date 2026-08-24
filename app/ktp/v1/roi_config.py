import cv2
import numpy as np
from app.core.logging_config import ktp_logger as logger

# Rasio standar KTP Indonesia (ID-1)
# Lebar: 85.6 mm, Tinggi: 53.98 mm
ID1_ASPECT_RATIO = 85.6 / 53.98


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_perspective_ktp(image: np.ndarray, pts: np.ndarray, target_width: int = 1000) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    target_height = int(target_width / ID1_ASPECT_RATIO)

    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    return warped


def normalize_canvas_v1(image: np.ndarray, target_width: int = 1000) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]

    # --- 1. Deteksi Kontur 4 Sudut KTP ---
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            img_area = h * w

            for c in contours[:3]:
                area = cv2.contourArea(c)
                # Guard-Rail 1: Contour must cover > 35% of total image area
                if area < 0.35 * img_area:
                    continue

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    rect_pts = order_points(pts)
                    (tl, tr, br, bl) = rect_pts
                    w_cand = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
                    h_cand = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
                    
                    if h_cand > 0:
                        aspect_cand = float(w_cand / h_cand)
                        # Guard-Rail 2: Aspect ratio must be close to ID-1 KTP ratio (85.6/53.98 = 1.586)
                        if not (1.40 <= aspect_cand <= 1.80):
                            logger.info(f"[ROI DEBUG] Contour rejected due to invalid aspect ratio: {aspect_cand:.2f}")
                            continue

                    warped = warp_perspective_ktp(image, pts, target_width=target_width)
                    logger.info("[ROI DEBUG] 4-Point KTP Perspective Warp Successful!")
                    return warped
    except Exception as e:
        logger.warning(f"[ROI DEBUG] Contour warp exception: {str(e)}")

    # --- 2. Fallback: Padding Aspect Ratio ID-1 ---
    current_ratio = w / h
    if abs(current_ratio - ID1_ASPECT_RATIO) > 0.05:
        if current_ratio > ID1_ASPECT_RATIO:
            new_h = int(w / ID1_ASPECT_RATIO)
            pad_h = new_h - h
            top = pad_h // 2
            bottom = pad_h - top
            image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            new_w = int(h * ID1_ASPECT_RATIO)
            pad_w = new_w - w
            left = pad_w // 2
            right = pad_w - left
            image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    target_height = int(target_width / ID1_ASPECT_RATIO)
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_image

ROI_CONFIG = {
    "nik":               {"x_min": 0.15, "y_min": 0.17, "x_max": 0.95, "y_max": 0.25},
    "nama":              {"x_min": 0.15, "y_min": 0.25, "x_max": 0.75, "y_max": 0.31},
    "tempat_lahir":      {"x_min": 0.15, "y_min": 0.31, "x_max": 0.75, "y_max": 0.37},
    "tanggal_lahir":     {"x_min": 0.15, "y_min": 0.31, "x_max": 0.75, "y_max": 0.37},
    "jenis_kelamin":     {"x_min": 0.15, "y_min": 0.37, "x_max": 0.55, "y_max": 0.42},
    "golongan_darah":    {"x_min": 0.55, "y_min": 0.37, "x_max": 0.78, "y_max": 0.42},
    "alamat":            {"x_min": 0.15, "y_min": 0.42, "x_max": 0.85, "y_max": 0.46},
    "rt_rw":             {"x_min": 0.15, "y_min": 0.46, "x_max": 0.60, "y_max": 0.50},
    "kelurahan_desa":    {"x_min": 0.15, "y_min": 0.50, "x_max": 0.65, "y_max": 0.54},
    "kecamatan":         {"x_min": 0.15, "y_min": 0.54, "x_max": 0.65, "y_max": 0.58},
    "agama":             {"x_min": 0.15, "y_min": 0.58, "x_max": 0.55, "y_max": 0.62},
    "status_perkawinan": {"x_min": 0.15, "y_min": 0.62, "x_max": 0.65, "y_max": 0.66},
    "pekerjaan":         {"x_min": 0.15, "y_min": 0.66, "x_max": 0.75, "y_max": 0.71},
    "kewarganegaraan":   {"x_min": 0.15, "y_min": 0.71, "x_max": 0.52, "y_max": 0.76},
    "berlaku_hingga":    {"x_min": 0.52, "y_min": 0.71, "x_max": 0.95, "y_max": 0.76},
}

WHITELISTS = {
    "nik": "",
    "nama": "",
    "tempat_lahir": "",
    "tanggal_lahir": "",
    "jenis_kelamin": "",
    "golongan_darah": "",
    "alamat": "",
    "rt_rw": "",
    "kelurahan_desa": "",
    "kecamatan": "",
    "agama": "",
    "status_perkawinan": "",
    "pekerjaan": "",
    "kewarganegaraan": "",
    "berlaku_hingga": ""
}
