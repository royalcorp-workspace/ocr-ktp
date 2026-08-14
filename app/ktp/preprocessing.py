import cv2
import numpy as np
import pytesseract
from app.core.logging_config import ktp_logger as logger


def deskew(gray: np.ndarray) -> np.ndarray:
    """Koreksi kemiringan foto KTP. Kegagalan tidak boleh menjatuhkan pipeline."""
    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))

        if coords.shape[0] < 50:
            return gray

        coords = coords.astype(np.float32)
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5 or abs(angle) > 15:
            return gray

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception as e:
        logger.warning(f"Deskew gagal, lanjut tanpa koreksi kemiringan: {e}")
        return gray


def hough_deskew(gray: np.ndarray) -> np.ndarray:
    """Deskew berbasis Hough Lines untuk kemiringan hingga +/-45 derajat."""
    try:
        (h, w) = gray.shape[:2]
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=w // 4, maxLineGap=20)
        if lines is None:
            return gray

        angles = []
        for l in lines:
            line = l.ravel()
            if len(line) == 4:
                x1, y1, x2, y2 = line
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                if abs(angle) < 45:
                    angles.append(angle)
                elif abs(angle - 90) < 45:
                    angles.append(angle - 90)
                elif abs(angle + 90) < 45:
                    angles.append(angle + 90)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5 or abs(median_angle) > 45:
            return gray

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        logger.warning(f"Hough deskew failed: {e}")
        return gray


def soft_unsharp_mask(gray: np.ndarray) -> np.ndarray:
    """Penajaman kontras 2D yang lembut tanpa efek garis ganda atau merusak piksel angka."""
    gaussian = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.addWeighted(gray, 1.25, gaussian, -0.25, 0)


def sharpen_horizontal_motion(gray: np.ndarray) -> np.ndarray:
    """Sharpening khusus sumbu horizontal untuk mengatasi motion blur geser horizontal."""
    kernel_h = np.array([[-1, -1, 5, -1, -1]], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, kernel_h)
    return sharpened


def adaptive_gamma_correction(gray: np.ndarray) -> np.ndarray:
    """Gamma correction adaptif berdasarkan mean brightness."""
    mean_val = float(np.mean(gray))
    if mean_val < 110:
        gamma = 1.6
    elif mean_val > 180:
        gamma = 0.7
    else:
        gamma = 1.3

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(gray, table)


def auto_orient_image(image: np.ndarray) -> np.ndarray:
    """Deteksi & koreksi rotasi 0/90/180/270 derajat pakai Tesseract OSD."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        osd = pytesseract.image_to_osd(gray, config="-c omp_thread_limit=1")
        rotate_angle = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                rotate_angle = int(line.split(":")[1].strip())
                break

        if rotate_angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif rotate_angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        logger.info(f"OSD detection skipped or inconclusive: {e}")

    return image


def resize_if_small(image: np.ndarray, target_width: int = 1600) -> np.ndarray:
    """Perbesar gambar jika lebih kecil dari target_width, pakai LANCZOS4."""
    height, width = image.shape[:2]
    if width < target_width:
        scale_factor = target_width / width
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image


def normalize_image_size(image: np.ndarray, target_width: int = 1600, max_width: int = 1800) -> np.ndarray:
    """
    Normalisasi ukuran gambar KTP:
    - Jika width > max_width (kamera HP 4K/8K): downscale ke max_width menggunakan INTER_AREA.
    - Jika width < target_width (gambar kecil/crop): upscale ke target_width menggunakan INTER_LANCZOS4.
    """
    height, width = image.shape[:2]
    if width > max_width:
        scale_factor = max_width / width
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    elif width < target_width:
        scale_factor = target_width / width
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image


def build_tier1_candidates(image: np.ndarray) -> list:
    """
    Tier 1: 5 Kandidat tercepat dan memprioritaskan piksel grayscale alami gambar.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = deskew(gray)

    clahe_soft = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_soft = clahe_soft.apply(gray)

    # 1. Red Channel Optimization (best contrast for black text on blue BG)
    red_ch = image[:, :, 2] if len(image.shape) == 3 else gray
    red_enhanced = clahe_soft.apply(red_ch)

    unsharp_soft = soft_unsharp_mask(enhanced_soft)

    # 2. Thickened Grayscale (Erosion) to reconnect broken digit waists (e.g. '8' -> '4')
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thick_gray = cv2.erode(gray, kernel_erode, iterations=1)
    thick_enhanced = clahe_soft.apply(thick_gray)

    # 3. BG Removal TopHat
    w_img = gray.shape[1]
    k_width = max(15, min(50, w_img // 40))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (k_width, 1))
    morph_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_bg)
    bg_removed = cv2.absdiff(gray, morph_open)
    bg_removed = cv2.bitwise_not(bg_removed)
    bilateral_bg = cv2.bilateralFilter(bg_removed, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian_bg = cv2.GaussianBlur(bilateral_bg, (0, 0), 2.0)
    unsharp_bg = cv2.addWeighted(bilateral_bg, 1.5, gaussian_bg, -0.5, 0)

    return [
        ("Pure Grayscale (PSM 6)", gray, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Soft CLAHE Grayscale (PSM 6)", enhanced_soft, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Thickened Grayscale (PSM 6)", thick_enhanced, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Red Channel CLAHE (PSM 6)", red_enhanced, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Soft Unsharp Mask (PSM 6)", unsharp_soft, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("BG Removal TopHat Mild (PSM 6)", unsharp_bg, "--oem 3 --psm 6 -c omp_thread_limit=1"),
    ]


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Normalisasi pencahayaan adaptif untuk gambar yang memiliki bayangan atau gelap."""
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        diff = cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return cv2.bitwise_not(norm)
    except Exception as e:
        logger.warning(f"Illumination normalization failed: {e}")
        return gray


def build_tier2_candidates(image: np.ndarray) -> list:
    """
    Tier 2: Fallback candidates untuk gambar yang sulit/memiliki distorsi warna atau noise berat.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = deskew(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    clahe_aggr = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_aggr = clahe_aggr.apply(gray)

    # Illumination Normalization
    gray_illum = normalize_illumination(gray)
    enhanced_illum = clahe.apply(gray_illum)

    w_img = gray.shape[1]
    k_width = max(15, min(50, w_img // 40))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (k_width, 1))

    # HSV Filter
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 20, 20])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        image_no_blue = image.copy()
        image_no_blue[blue_mask > 0] = [255, 255, 255]
        gray_no_blue = cv2.cvtColor(image_no_blue, cv2.COLOR_BGR2GRAY)

        inpainted_img = cv2.inpaint(image, blue_mask, 3, cv2.INPAINT_TELEA)
        gray_inpainted = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_no_blue = gray.copy()
        gray_inpainted = gray.copy()

    morph_hsv = cv2.morphologyEx(gray_no_blue, cv2.MORPH_OPEN, kernel_bg)
    bg_removed_hsv = cv2.absdiff(gray_no_blue, morph_hsv)
    bg_removed_hsv = cv2.bitwise_not(bg_removed_hsv)
    bilateral_hsv = cv2.bilateralFilter(bg_removed_hsv, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian_hsv = cv2.GaussianBlur(bilateral_hsv, (0, 0), 2.0)
    unsharp_hsv = cv2.addWeighted(bilateral_hsv, 1.5, gaussian_hsv, -0.5, 0)

    # Hough Deskew
    gray_raw_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_hough_deskewed = hough_deskew(gray_raw_full)
    hough_clahe = clahe.apply(gray_hough_deskewed)

    # Adaptive Gamma
    gray_gamma = adaptive_gamma_correction(gray_raw_full)
    gray_gamma_deskewed = hough_deskew(gray_gamma)
    gray_gamma_clahe = clahe.apply(gray_gamma_deskewed)
    gaussian_gamma = cv2.GaussianBlur(gray_gamma_clahe, (0, 0), 2.0)
    unsharp_gamma = cv2.addWeighted(gray_gamma_clahe, 1.8, gaussian_gamma, -0.8, 0)

    # Adaptive Threshold C10 & Median Blur
    denoised_med = cv2.medianBlur(enhanced, 3)
    _, otsu_med = cv2.threshold(denoised_med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_c10 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    return [
        ("Illumination Norm + CLAHE (PSM 6)", enhanced_illum, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Inpaint Blue Mask (PSM 6)", gray_inpainted, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("HSV Filter + TopHat + Unsharp (PSM 6)", unsharp_hsv, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Hough Deskew + CLAHE (PSM 6)", hough_clahe, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Adaptive Gamma + Hough Deskew + Unsharp (PSM 6)", unsharp_gamma, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("CLAHE Aggressive Grayscale (PSM 6)", enhanced_aggr, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("CLAHE + Median Blur + Otsu (PSM 6)", otsu_med, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("CLAHE + Adaptive Gaussian C10 (PSM 6)", adaptive_c10, "--oem 3 --psm 6 -c omp_thread_limit=1"),
    ]


def build_tier3_candidates(image: np.ndarray) -> list:
    """
    Tier 3: Rotation Fallback (90, 180, 270 derajat). HANYA dieksekusi jika Tier 1 & 2 gagal total.
    """
    img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    img_180 = cv2.rotate(image, cv2.ROTATE_180)
    img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    gray_90 = cv2.cvtColor(img_90, cv2.COLOR_BGR2GRAY) if len(img_90.shape) == 3 else img_90
    gray_180 = cv2.cvtColor(img_180, cv2.COLOR_BGR2GRAY) if len(img_180.shape) == 3 else img_180
    gray_270 = cv2.cvtColor(img_270, cv2.COLOR_BGR2GRAY) if len(img_270.shape) == 3 else img_270

    red_90 = img_90[:, :, 2] if len(img_90.shape) == 3 else gray_90
    red_180 = img_180[:, :, 2] if len(img_180.shape) == 3 else gray_180
    red_270 = img_270[:, :, 2] if len(img_270.shape) == 3 else gray_270

    red_enh_90 = clahe.apply(red_90)
    red_enh_180 = clahe.apply(red_180)
    red_enh_270 = clahe.apply(red_270)

    return [
        ("Rotated 90 Red CLAHE (PSM 6)", red_enh_90, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Rotated 90 Pure (PSM 6)", gray_90, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Rotated 180 Red CLAHE (PSM 6)", red_enh_180, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Rotated 180 Pure (PSM 6)", gray_180, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Rotated 270 Red CLAHE (PSM 6)", red_enh_270, "--oem 3 --psm 6 -c omp_thread_limit=1"),
        ("Rotated 270 Pure (PSM 6)", gray_270, "--oem 3 --psm 6 -c omp_thread_limit=1"),
    ]


def build_candidates(image: np.ndarray) -> list:
    """Fallback kompatibilitas: mengembalikan seluruh kandidat menggabungkan Tier 1, 2, dan 3."""
    return build_tier1_candidates(image) + build_tier2_candidates(image) + build_tier3_candidates(image)