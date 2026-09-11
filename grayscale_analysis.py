"""Grayscale forensic branch used as supporting evidence, not a classifier."""
import cv2
import numpy as np

def grayscale_inconsistency_map(image_bgr):
    """Find locally unusual luminance/noise structure at multiple scales."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine_residual = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.0))
    coarse_residual = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 5.0))
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    local_noise = cv2.GaussianBlur(fine_residual ** 2, (0, 0), 7) ** .5
    def normalize(data): return cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)
    return .30 * normalize(fine_residual) + .25 * normalize(coarse_residual) + .30 * normalize(laplacian) + .15 * normalize(local_noise)

def analyze_grayscale(image_bgr):
    evidence = grayscale_inconsistency_map(image_bgr)
    return round(float(np.clip(np.mean(evidence) * 150, 0, 100)), 1)
