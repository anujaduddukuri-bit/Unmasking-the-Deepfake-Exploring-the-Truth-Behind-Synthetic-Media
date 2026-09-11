"""Build cautious visual evidence maps from local colour and frequency residuals."""
import cv2
import numpy as np
from grayscale_analysis import grayscale_inconsistency_map

def suspiciousness_map(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    colour_residual = np.linalg.norm(lab - cv2.GaussianBlur(lab, (0, 0), 5), axis=2)
    high_frequency = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.5))
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    grayscale_evidence = grayscale_inconsistency_map(image_bgr)
    def norm(value): return cv2.normalize(value, None, 0, 1, cv2.NORM_MINMAX)
    # Grayscale/noise evidence carries most weight so brightness and resampling artifacts
    # are visible even where colour is uniform.
    combined = .20*norm(colour_residual) + .20*norm(high_frequency) + .15*norm(laplacian) + .45*grayscale_evidence
    combined = cv2.GaussianBlur(combined, (0, 0), 2)
    return cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def generate_visualizations(image_bgr):
    score_map = suspiciousness_map(image_bgr)
    heatmap = cv2.applyColorMap(score_map, cv2.COLORMAP_JET) # blue low, red high
    heatmap_overlay = cv2.addWeighted(image_bgr, .45, heatmap, .55, 0)
    return heatmap_overlay, score_map

