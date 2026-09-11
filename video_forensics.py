"""Temporal video forensics: inter-frame flickering, noise jitter, and temporal consistency."""
import cv2
import numpy as np

def compute_temporal_metrics(frames_bgr, per_frame_metrics):
    """Analyze temporal consistency across sequential video frames.
    
    Args:
        frames_bgr: list of BGR numpy arrays for each extracted frame.
        per_frame_metrics: list of dicts with frame scores (suspicion, noise, color, grayscale).
        
    Returns:
        dict containing video-level forensic scores, jitter indicators, and timeline data.
    """
    if not frames_bgr or len(frames_bgr) < 2:
        return {
            "temporal_inconsistency_score": 0.0,
            "temporal_jitter_score": 0.0,
            "color_drift_score": 0.0,
            "peak_frame": per_frame_metrics[0]["number"] if per_frame_metrics else 1,
            "peak_score": per_frame_metrics[0]["suspicion_score"] if per_frame_metrics else 0.0,
            "timeline": []
        }

    frame_diffs = []
    color_diffs = []
    
    # Analyze adjacent frame pairs
    for i in range(1, len(frames_bgr)):
        prev_gray = cv2.cvtColor(frames_bgr[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Resize to standard 224x224 for uniform metric evaluation
        if prev_gray.shape != (224, 224):
            prev_gray = cv2.resize(prev_gray, (224, 224))
            curr_gray = cv2.resize(curr_gray, (224, 224))
            
        # Absolute frame-to-frame luminance difference
        diff = np.mean(np.abs(curr_gray - prev_gray))
        frame_diffs.append(float(diff))
        
        # Color balance change between frames
        prev_hsv = cv2.cvtColor(cv2.resize(frames_bgr[i - 1], (224, 224)), cv2.COLOR_BGR2HSV).astype(np.float32)
        curr_hsv = cv2.cvtColor(cv2.resize(frames_bgr[i], (224, 224)), cv2.COLOR_BGR2HSV).astype(np.float32)
        c_diff = np.mean(np.abs(curr_hsv[:, :, 0] - prev_hsv[:, :, 0])) / 180.0
        color_diffs.append(float(c_diff))

    # Noise jitter: standard deviation of noise scores across sequence
    noise_scores = [m["noise_score"] for m in per_frame_metrics]
    noise_variance = float(np.std(noise_scores))
    
    # Suspicion metrics
    suspicion_scores = [m["suspicion_score"] for m in per_frame_metrics]
    avg_suspicion = float(np.mean(suspicion_scores))
    peak_idx = int(np.argmax(suspicion_scores))
    peak_frame = per_frame_metrics[peak_idx]["number"]
    peak_score = suspicion_scores[peak_idx]
    peak_timestamp = per_frame_metrics[peak_idx].get("timestamp", 0.0)

    # Normalized score components
    diff_norm = np.clip(np.mean(frame_diffs) / 45.0, 0.0, 1.0)
    jitter_norm = np.clip(noise_variance / 20.0, 0.0, 1.0)
    drift_norm = np.clip(np.mean(color_diffs) * 3.0, 0.0, 1.0)

    # Composite temporal inconsistency score (0 - 100%)
    temporal_inconsistency = round(float((0.40 * diff_norm + 0.35 * jitter_norm + 0.25 * drift_norm) * 100), 1)
    temporal_jitter = round(float(jitter_norm * 100), 1)
    color_drift = round(float(drift_norm * 100), 1)

    # Prepare timeline data for frontend charting
    timeline = []
    for i, m in enumerate(per_frame_metrics):
        timeline.append({
            "frame": m["number"],
            "timestamp": m.get("timestamp", round(i * 0.5, 2)),
            "suspicion": m["suspicion_score"],
            "noise": m["noise_score"],
            "color": m["color_score"],
            "grayscale": m["grayscale_score"],
            "diff_from_prev": round(frame_diffs[i - 1], 2) if i > 0 else 0.0
        })

    return {
        "temporal_inconsistency_score": temporal_inconsistency,
        "temporal_jitter_score": temporal_jitter,
        "color_drift_score": color_drift,
        "average_suspicion": round(avg_suspicion, 1),
        "peak_frame": peak_frame,
        "peak_score": peak_score,
        "peak_timestamp": peak_timestamp,
        "timeline": timeline
    }
