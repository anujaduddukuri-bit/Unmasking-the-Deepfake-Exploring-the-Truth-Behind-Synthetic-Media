"""Supporting colour-forensics signal; never a standalone deepfake verdict."""
import cv2
import numpy as np

def _normalize(value, low, high): return float(np.clip((value - low) / (high - low + 1e-8), 0, 1))

def analyze_color(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    channel_means = rgb.mean(axis=(0, 1)); balance = np.std(channel_means) / (np.mean(channel_means) + 1e-8)
    saturation = hsv[:, :, 1].astype(np.float32); brightness = hsv[:, :, 2].astype(np.float32)
    local_sat = cv2.GaussianBlur(saturation, (0, 0), 7)
    local_lab = cv2.GaussianBlur(lab.astype(np.float32), (0, 0), 7)
    local_variation = np.mean(np.abs(saturation - local_sat)) / 255
    lab_residual = np.mean(np.linalg.norm(lab.astype(np.float32) - local_lab, axis=2)) / 181
    transitions = np.mean(cv2.Laplacian(saturation, cv2.CV_32F).__abs__()) / 255
    exposure = abs(brightness.mean() - 127.5) / 127.5
    raw = (.25*_normalize(balance, .02, .35) + .25*np.clip(local_variation*4,0,1) +
           .25*np.clip(lab_residual*4,0,1) + .15*np.clip(transitions*3,0,1) + .10*exposure)
    return round(float(np.clip(raw * 100, 0, 100)), 1)
