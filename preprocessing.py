"""Image and video-ready preprocessing functions."""
import math
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def inference_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def preprocess_pil(image):
    return inference_transform()(image.convert("RGB"))

def preprocess_frames(frames):
    """Convert RGB PIL images into [batch, sequence, channels, height, width]."""
    if not frames: raise ValueError("No frames were supplied.")
    return torch.stack([preprocess_pil(frame) for frame in frames]).unsqueeze(0)

def extract_video_frames(video_path, max_frames=30, max_fps=30):
    """Extract individual RGB frames uniformly across the video at up to 30 FPS.
    
    Returns:
        frames: list of PIL Images (RGB)
        timestamps: list of timestamps in seconds
        sampled_fps: effective sampling rate in FPS (capped at 30)
        source_fps: native FPS of the source video
        duration: total video duration in seconds
    """
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or not capture.isOpened():
        capture.release()
        raise ValueError("Could not open or read video frames.")
    
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if source_fps <= 0 or math.isnan(source_fps):
        source_fps = 30.0
    
    duration = total_frames / source_fps
    
    # Cap effective fps at max_fps (30)
    effective_fps = min(source_fps, float(max_fps))
    
    # Calculate frame indices to sample uniformly across the video up to max_frames
    num_to_sample = min(total_frames, max_frames)
    if num_to_sample <= 1:
        target_indices = [0]
    else:
        # Uniform spacing across entire video timeline
        target_indices = [int(round(i * (total_frames - 1) / (num_to_sample - 1))) for i in range(num_to_sample)]
        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for idx in target_indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        target_indices = deduped
    
    frames, timestamps = [], []
    current_idx = 0
    target_set = set(target_indices)
    
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if current_idx in target_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(round(current_idx / source_fps, 2))
            if len(frames) == len(target_indices):
                break
        current_idx += 1
        
    capture.release()
    if not frames:
        raise ValueError("No readable video frames could be extracted.")
        
    sampled_fps = min(effective_fps, round(len(frames) / max(duration, 1.0), 2))
    return frames, timestamps, sampled_fps, source_fps, round(duration, 2)
