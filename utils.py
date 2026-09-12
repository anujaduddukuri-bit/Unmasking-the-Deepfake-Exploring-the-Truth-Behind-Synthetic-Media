"""Shared paths, seed setup, and safe filesystem helpers."""
import os
import tempfile
from pathlib import Path
import random
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

def _get_writable_dir(subfolder):
    primary = BASE_DIR / subfolder
    try:
        primary.mkdir(parents=True, exist_ok=True)
        test_file = primary / ".write_test"
        test_file.touch()
        test_file.unlink()
        return primary
    except (OSError, PermissionError):
        tmp = Path(tempfile.gettempdir()) / "unmasking_deepfake" / subfolder
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = _get_writable_dir("results")
HEATMAPS_DIR = _get_writable_dir(Path("results") / "heatmaps")
VIDEO_FRAMES_DIR = _get_writable_dir(Path("results") / "video_frames")
REPORTS_DIR = _get_writable_dir(Path("results") / "reports")
UPLOADS_DIR = _get_writable_dir(Path("static") / "uploads")

MODEL_PATH = MODELS_DIR / "deepfake_model.pth"
ONNX_MODEL_PATH = MODELS_DIR / "deepfake_model.onnx"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

def ensure_directories():
    for path in (MODELS_DIR, HEATMAPS_DIR, VIDEO_FRAMES_DIR, REPORTS_DIR, UPLOADS_DIR):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"mp4", "avi", "mov", "mkv", "webm"}
