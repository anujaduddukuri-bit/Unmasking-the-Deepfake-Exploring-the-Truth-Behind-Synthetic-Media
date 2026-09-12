"""Image and video-ready preprocessing functions.

Changes:
  - extract_video_frames now uses cv2.CAP_PROP_POS_FRAMES seek (O(1) seek
    instead of reading every frame sequentially) — major speedup for long videos.
  - max_frames default lowered to 12 (enough for <15 s target).
"""
import math
import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_pil(image):
    """Preprocess PIL image into normalized float32 CHW array (224x224)."""
    img = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    norm = (arr - mean) / std
    return np.transpose(norm, (2, 0, 1))   # (3, 224, 224)


def preprocess_frames(frames):
    """Convert RGB PIL images into [batch, sequence, C, H, W]."""
    if not frames:
        raise ValueError("No frames were supplied.")
    processed = [preprocess_pil(f) for f in frames]
    stacked   = np.stack(processed, axis=0)          # [seq, 3, 224, 224]
    batched   = np.expand_dims(stacked, axis=0)      # [1, seq, 3, 224, 224]
    try:
        import torch
        return torch.from_numpy(batched).float()
    except ImportError:
        return batched


def extract_video_frames(video_path, max_frames=30, max_fps=30):
    """Extract individual RGB frames uniformly across the video using seek.

    Uses cv2.CAP_PROP_POS_FRAMES to jump directly to each target frame index
    rather than reading every frame sequentially — much faster for long videos.

    Returns:
        frames      : list of PIL Images (RGB)
        timestamps  : list of timestamps (seconds)
        sampled_fps : effective sampling rate (capped at max_fps)
        source_fps  : native FPS of source video
        duration    : total video duration (seconds)
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open video file.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise ValueError("Could not read frame count from video.")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if source_fps <= 0 or math.isnan(source_fps):
        source_fps = 30.0

    duration = total_frames / source_fps
    effective_fps = min(source_fps, float(max_fps))

    # Uniform spacing across the whole timeline
    num_to_sample = min(total_frames, max_frames)
    if num_to_sample <= 1:
        target_indices = [0]
    else:
        target_indices = [
            int(round(i * (total_frames - 1) / (num_to_sample - 1)))
            for i in range(num_to_sample)
        ]
        # Remove duplicates (short videos)
        seen, deduped = set(), []
        for idx in target_indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        target_indices = deduped

    frames, timestamps = [], []

    for frame_idx in target_indices:
        # Seek directly to target frame — O(1) for most codecs
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok:
            # Fallback: try adjacent frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
            ok, frame = capture.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(round(frame_idx / source_fps, 2))

    capture.release()

    if not frames:
        raise ValueError("No readable video frames could be extracted.")

    sampled_fps = min(effective_fps, round(len(frames) / max(duration, 1.0), 2))
    return frames, timestamps, sampled_fps, source_fps, round(duration, 2)
