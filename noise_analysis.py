"""Forensic noise analysis: Sand-Pour Noise Blueprint and Sand Noise Overlay algorithms."""
import cv2
import numpy as np

def extract_sand_noise_residual(image_bgr):
    """Extract fine multi-scale high-frequency noise residuals and local variance."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Scale 1: Micro-residual using median filter difference (fine sensor grain & tampering seams)
    median_blur = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    micro_residual = np.abs(gray - median_blur)
    
    # Scale 2: Gaussian high-frequency residual
    gauss_blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gauss_residual = np.abs(gray - gauss_blur)
    
    # Scale 3: Laplacian high-frequency edge & texture response
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    
    # Multi-scale fusion
    norm = lambda x: cv2.normalize(x, None, 0.0, 1.0, cv2.NORM_MINMAX)
    fused = 0.40 * norm(micro_residual) + 0.35 * norm(gauss_residual) + 0.25 * norm(laplacian)
    
    # Local noise variance in 5x5 neighbourhood
    local_mean = cv2.boxFilter(fused, cv2.CV_32F, (5, 5))
    local_sq = cv2.boxFilter(fused ** 2, cv2.CV_32F, (5, 5))
    local_var = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0.0))
    
    # Generate granular sand micro-texture deterministically from pixel coordinates and residual
    h, w = gray.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    pseudo_grain = np.sin(x_coords * 12.9898 + y_coords * 78.233) * 43758.5453
    pseudo_grain = (pseudo_grain - np.floor(pseudo_grain)).astype(np.float32)
    
    # Modulate fused noise with granular sand texture
    sand_energy = fused * 0.75 + local_var * 0.45 + (fused * pseudo_grain) * 0.35
    return np.clip(sand_energy, 0.0, 1.0)

def sand_noise_blueprint(image_bgr):
    """Render an architectural blueprint of the image identified by noise like poured sand.
    
    The canvas is a dark midnight Prussian navy blue (#030a16). High-frequency noise
    and structural contours emerge as glowing cyan, azure, and white sand grains,
    exposing synthetic artifacts and compression boundaries.
    """
    sand_energy = extract_sand_noise_residual(image_bgr)
    norm_energy = cv2.normalize(sand_energy, None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    h, w = image_bgr.shape[:2]
    blueprint = np.zeros((h, w, 3), dtype=np.uint8)
    
    val = norm_energy[:, :, None]
    
    # Color palette definition in BGR:
    # Deep Navy Base:   B=26,  G=12,  R=4
    # Slate Cerulean:   B=125, G=75,  R=18
    # Electric Cyan:    B=255, G=210, R=20
    # Starlight White:  B=255, G=255, R=255
    layer1 = np.array([26, 12, 4], dtype=np.float32)       # Base navy
    layer2 = np.array([125, 75, 18], dtype=np.float32)     # Cerulean shadow
    layer3 = np.array([255, 210, 20], dtype=np.float32)    # Electric cyan grains
    layer4 = np.array([255, 255, 255], dtype=np.float32)   # Hotspot sand crests
    
    t1 = np.clip(val / 0.30, 0.0, 1.0)
    t2 = np.clip((val - 0.30) / 0.35, 0.0, 1.0)
    t3 = np.clip((val - 0.65) / 0.35, 0.0, 1.0)
    
    color = layer1 * (1.0 - t1) + layer2 * t1
    color = color * (1.0 - t2) + layer3 * t2
    color = color * (1.0 - t3) + layer4 * t3
    
    return np.clip(color, 0, 255).astype(np.uint8)

# Backward-compatible alias
noise_blueprint = sand_noise_blueprint


def analyze_noise(image_bgr):
    """Forensic noise metric combining multi-scale residuals, Laplacian variance, and blockiness."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_32F).var()
    local_mean = cv2.blur(residual, (25, 25))
    local_sq = cv2.blur(residual ** 2, (25, 25))
    local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    variation = local_std.std() / (local_std.mean() + 1e-6)
    blockiness = (np.abs(np.diff(gray, axis=1)[:, 7::8]).mean() + np.abs(np.diff(gray, axis=0)[7::8, :]).mean()) / 2
    
    sand_energy = extract_sand_noise_residual(image_bgr)
    sand_intensity = float(np.mean(sand_energy))
    
    raw = (0.30 * np.clip(variation / 1.2, 0, 1) +
           0.25 * np.clip(laplacian_variance / 1200, 0, 1) +
           0.25 * np.clip(blockiness / 25, 0, 1) +
           0.20 * np.clip(sand_intensity * 1.5, 0, 1))
    return round(float(np.clip(raw * 100, 0, 100)), 1)
