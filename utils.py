"""Shared paths, seed setup, and safe filesystem helpers."""
from pathlib import Path
import random
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"
VIDEO_FRAMES_DIR = RESULTS_DIR / "video_frames"
REPORTS_DIR = RESULTS_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = MODELS_DIR / "deepfake_model.pth"
ONNX_MODEL_PATH = MODELS_DIR / "deepfake_model.onnx"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

def ensure_directories():
    for path in (MODELS_DIR, HEATMAPS_DIR, VIDEO_FRAMES_DIR, REPORTS_DIR, UPLOADS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR == (BASE_DIR / "dataset"):
        (DATASET_DIR / "real").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "fake").mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"mp4", "avi", "mov", "mkv", "webm"}
