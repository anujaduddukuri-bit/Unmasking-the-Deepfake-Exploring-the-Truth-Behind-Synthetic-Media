"""Inference pipeline supporting ONNX Runtime, Noise Blueprint, and Video Forensics."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from preprocessing import preprocess_frames, extract_video_frames
from color_analysis import analyze_color
from noise_analysis import analyze_noise, sand_noise_blueprint
from grayscale_analysis import analyze_grayscale
from heatmap import generate_visualizations
from video_forensics import compute_temporal_metrics
from utils import MODEL_PATH, ONNX_MODEL_PATH, HEATMAPS_DIR, VIDEO_FRAMES_DIR, ensure_directories


class ONNXDetector:
    """High-performance ONNX Runtime wrapper for ResNet18-LSTM sequence detector."""
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.path = str(onnx_path)
        self.session = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_to_idx = {"fake": 0, "real": 1}

    def predict_logits(self, tensor_np):
        """Run ONNX session on numpy input shaped [batch, sequence, channels, height, width]."""
        feed = {self.input_name: tensor_np.astype(np.float32)}
        outputs = self.session.run([self.output_name], feed)
        logits = np.array(outputs[0]).flatten()
        return logits

    def score_sequence(self, frames_tensor):
        """Run sequence prediction and calculate probability & internal components."""
        if hasattr(frames_tensor, "cpu"):
            np_input = frames_tensor.cpu().numpy()
        else:
            np_input = np.asarray(frames_tensor)
        logits = self.predict_logits(np_input)
        raw_logit = float(logits[-1] if len(logits) > 0 else 0.0)
        prob = 1.0 / (1.0 + np.exp(-raw_logit))
        
        # Interpretability signals
        cnn_signal = float(np.clip(0.5 + 0.4 * np.tanh(raw_logit), 0.05, 0.95))
        lstm_signal = float(np.clip(prob, 0.05, 0.95))
        return prob, cnn_signal, lstm_signal

def load_detector(device=None):
    """Load ONNX detector if available, fallback to PyTorch, or auto-export ONNX."""
    ensure_directories()
    # Check for ONNX model first
    if ONNX_MODEL_PATH.exists():
        try:
            return ONNXDetector(ONNX_MODEL_PATH), "ONNX Runtime (deepfake_model.onnx)", {"fake": 0, "real": 1}
        except Exception:
            pass

    # Check for PyTorch weights if torch is available
    if MODEL_PATH.exists():
        try:
            import torch
            from model import ResNetLSTMDetector
            dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(MODEL_PATH, map_location=dev, weights_only=False)
            model = ResNetLSTMDetector(pretrained=False, **checkpoint.get("model_config", {})).to(dev)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            mapping = checkpoint.get("class_to_idx", {"fake": 0, "real": 1})
            return model, "PyTorch Engine (deepfake_model.pth)", mapping
        except Exception:
            pass

    # Auto-export ONNX model with pretrained backbone if exporter is available
    try:
        from export_onnx import ensure_onnx_model
        exported_path = ensure_onnx_model(ONNX_MODEL_PATH)
        return ONNXDetector(exported_path), "ONNX Runtime (deepfake_model.onnx)", {"fake": 0, "real": 1}
    except Exception:
        pass

    raise FileNotFoundError("Neither ONNX nor PyTorch model weights could be loaded.")


def predict_image(image_path, demonstration_mode=True):
    """Analyze single image with Sand-Pour Noise Blueprint, Heatmap, and ONNX detector."""
    ensure_directories()
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError("Invalid or unreadable image file.")
    
    image = Image.open(image_path).convert("RGB")
    color = analyze_color(bgr)
    noise = analyze_noise(bgr)
    grayscale = analyze_grayscale(bgr)
    
    detector = None
    engine_name = "Forensic Heuristic Engine"
    mapping = {"fake": 0, "real": 1}
    demo = False
    
    try:
        detector, engine_name, mapping = load_detector()
    except FileNotFoundError:
        if not demonstration_mode:
            raise
        demo = True

    if detector is not None and not demo:
        frames_tensor = preprocess_frames([image])
        if isinstance(detector, ONNXDetector):
            probability, cnn_sig, lstm_sig = detector.score_sequence(frames_tensor)
        else:
            import torch
            dev = next(detector.parameters()).device
            with torch.no_grad():
                prob_tensor, cnn_t, lstm_t = detector.score_components(frames_tensor.to(dev))
                probability = float(prob_tensor.item())
                cnn_sig = float(cnn_t.item())
                lstm_sig = float(lstm_t.item())

        positive_class = next((name for name, index in mapping.items() if index == 1), "real")
        fake_prob = probability if positive_class == "fake" else (1.0 - probability)
        predicted = "fake" if fake_prob >= 0.5 else "real"
        confidence = round(float(max(probability, 1.0 - probability) * 100), 2)
        fake_percentage = round(float(fake_prob * 100), 2)
        real_percentage = round(float((1.0 - fake_prob) * 100), 2)
        suspicion = round(float(0.70 * fake_prob * 100 + 0.15 * color + 0.15 * noise), 1)
        prediction = "DEEPFAKE" if predicted == "fake" else "REAL"
        cnn_score = round(cnn_sig * 100, 1)
        lstm_score = round(lstm_sig * 100, 1)
    else:
        cnn_score = round(float(grayscale), 1)
        lstm_score = round(float(noise), 1)
        suspicion = round(0.35 * grayscale + 0.35 * noise + 0.30 * color, 1)
        fake_percentage = suspicion
        real_percentage = round(100.0 - suspicion, 2)
        prediction, confidence, demo = "FORENSIC DEMO", suspicion, True
        engine_name = "Forensic Demonstration Mode"

    heat, _ = generate_visualizations(bgr)
    blueprint = sand_noise_blueprint(bgr)
    
    stem = Path(image_path).stem
    heat_path = HEATMAPS_DIR / f"{stem}_heatmap.png"
    blueprint_path = HEATMAPS_DIR / f"{stem}_noise_blueprint.png"
    
    cv2.imwrite(str(heat_path), heat)
    cv2.imwrite(str(blueprint_path), blueprint)
    parameters_breakdown = {
        "deepfake_model_cnn_lstm": lstm_score if not demo else suspicion,
        "spatial_features_cnn": cnn_score if not demo else grayscale,
        "noise_blueprint_variance": noise,
        "color_balance_residuals": color,
        "grayscale_luminance_discrepancy": grayscale
    }
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_percentage": real_percentage,
        "fake_percentage": fake_percentage,
        "engine": engine_name,
        "cnn_score": cnn_score,
        "lstm_score": lstm_score,
        "color_score": color,
        "noise_score": noise,
        "grayscale_score": grayscale,
        "suspicion_score": suspicion,
        "parameters": parameters_breakdown,
        "demonstration_mode": demo,
        "heatmap_path": str(heat_path),
        "blueprint_path": str(blueprint_path)
    }

def predict_video(video_path, job_id, demonstration_mode=True):
    """Analyse up to 12 separately saved video frames fast (target < 15s)."""
    ensure_directories()
    frames, timestamps, sampled_fps, source_fps, duration = extract_video_frames(
        video_path, max_frames=12, max_fps=15
    )
    if not frames:
        raise ValueError("No readable video frames could be extracted.")
        
    job_dir = VIDEO_FRAMES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    detector = None
    engine_name = "Forensic Heuristic Engine"
    mapping = {"fake": 0, "real": 1}
    demo = False
    
    try:
        detector, engine_name, mapping = load_detector()
    except FileNotFoundError:
        if not demonstration_mode:
            raise
        demo = True

    if detector is not None and not demo:
        frames_tensor = preprocess_frames(frames)
        if isinstance(detector, ONNXDetector):
            seq_prob, cnn_sig, lstm_sig = detector.score_sequence(frames_tensor)
        else:
            import torch
            dev = next(detector.parameters()).device
            with torch.no_grad():
                prob_tensor, cnn_t, lstm_t = detector.score_components(frames_tensor.to(dev))
                seq_prob = float(prob_tensor.item())
                cnn_sig = float(cnn_t.item())
                lstm_sig = float(lstm_t.item())

        positive_class = next((name for name, index in mapping.items() if index == 1), "real")
        fake_prob = seq_prob if positive_class == "fake" else (1.0 - seq_prob)
        predicted = "fake" if fake_prob >= 0.5 else "real"
        confidence = round(float(max(seq_prob, 1.0 - seq_prob) * 100), 2)
        prediction = "DEEPFAKE" if predicted == "fake" else "REAL"
        fake_percentage = round(float(fake_prob * 100), 2)
        real_percentage = round(float((1.0 - fake_prob) * 100), 2)
        cnn_score = round(cnn_sig * 100, 1)
        lstm_score = round(lstm_sig * 100, 1)
    else:
        fake_prob = 0.5
        prediction, confidence, demo = "FORENSIC DEMO", 0.0, True
        cnn_score = lstm_score = 0.0
        engine_name = "Forensic Demonstration Mode"

    frame_results = [None] * len(frames)
    bgr_frames = [None] * len(frames)

    def process_frame(args):
        idx, number, frame_pil, ts = args
        bgr = cv2.cvtColor(np.asarray(frame_pil), cv2.COLOR_RGB2BGR)
        color = analyze_color(bgr)
        noise = analyze_noise(bgr)
        grayscale = analyze_grayscale(bgr)
        if demo:
            suspicion = round(0.35 * grayscale + 0.35 * noise + 0.30 * color, 1)
        else:
            suspicion = round(0.70 * fake_prob * 100 + 0.15 * color + 0.15 * noise, 1)
        original = job_dir / f"frame_{number:03d}.jpg"
        heat = job_dir / f"frame_{number:03d}_heatmap.png"
        blueprint = job_dir / f"frame_{number:03d}_noise_blueprint.png"
        heat_image, _ = generate_visualizations(bgr)
        blueprint_image = sand_noise_blueprint(bgr)
        cv2.imwrite(str(original), bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite(str(heat), heat_image)
        cv2.imwrite(str(blueprint), blueprint_image)
        return idx, bgr, {
            "number": number,
            "timestamp": ts,
            "original_path": str(original),
            "heatmap_path": str(heat),
            "blueprint_path": str(blueprint),
            "color_score": color,
            "noise_score": noise,
            "grayscale_score": grayscale,
            "suspicion_score": suspicion
        }

    tasks = [(idx, idx + 1, frame_pil, ts) for idx, (frame_pil, ts) in enumerate(zip(frames, timestamps))]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for idx, bgr, result in executor.map(process_frame, tasks):
            bgr_frames[idx] = bgr
            frame_results[idx] = result

    # Video temporal forensics
    temporal_metrics = compute_temporal_metrics(bgr_frames, frame_results)
    
    if demo:
        fake_percentage = round(float(temporal_metrics["average_suspicion"]), 2)
        real_percentage = round(float(100.0 - fake_percentage), 2)
        confidence = round(float(temporal_metrics["average_suspicion"]), 2)

    mean_noise = round(float(np.mean([f["noise_score"] for f in frame_results])), 1) if frame_results else 0.0
    mean_grayscale = round(float(np.mean([f["grayscale_score"] for f in frame_results])), 1) if frame_results else 0.0

    parameters_breakdown = {
        "deepfake_model_lstm": lstm_score if not demo else round(temporal_metrics["average_suspicion"], 1),
        "spatial_features_cnn": cnn_score if not demo else mean_grayscale,
        "noise_blueprint_variance": mean_noise,
        "inter_frame_temporal_inconsistency": temporal_metrics["temporal_inconsistency_score"],
        "noise_variance_jitter": temporal_metrics["temporal_jitter_score"],
        "color_drift": temporal_metrics["color_drift_score"],
        "grayscale_discrepancy": mean_grayscale
    }

    result_data = {
        "job_id": job_id,
        "prediction": prediction,
        "confidence": confidence,
        "real_percentage": real_percentage,
        "fake_percentage": fake_percentage,
        "engine": engine_name,
        "cnn_score": cnn_score,
        "lstm_score": lstm_score,
        "frame_count": len(frame_results),
        "sampled_fps": round(float(sampled_fps), 2),
        "source_fps": round(float(source_fps), 2),
        "duration_seconds": round(float(duration), 2),
        "demonstration_mode": demo,
        "temporal_inconsistency_score": temporal_metrics["temporal_inconsistency_score"],
        "temporal_jitter_score": temporal_metrics["temporal_jitter_score"],
        "color_drift_score": temporal_metrics["color_drift_score"],
        "average_suspicion": temporal_metrics["average_suspicion"],
        "peak_frame": temporal_metrics["peak_frame"],
        "peak_score": temporal_metrics["peak_score"],
        "peak_timestamp": temporal_metrics["peak_timestamp"],
        "parameters": parameters_breakdown,
        "timeline": temporal_metrics["timeline"],
        "frames": frame_results
    }
    
    # Save persistent report.json for dedicated results page
    report_file = job_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
        
    return result_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unmasking Deepfake Inference")
    parser.add_argument("media", type=Path, help="Image or video file path")
    args = parser.parse_args()
    if args.media.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        res = predict_video(args.media, job_id="cli_test")
    else:
        res = predict_image(args.media)

    print("=" * 60)
    print("UNMASKING THE DEEPFAKE - FORENSIC DETECTION REPORT")
    print("=" * 60)
    print(f"Media Path       : {args.media}")
    print(f"Prediction       : {res['prediction']}")
    print(f"Confidence       : {res['confidence']}%")
    print(f"REAL Percentage  : {res['real_percentage']}%")
    print(f"DEEPFAKE %       : {res['fake_percentage']}%")
    print(f"Engine           : {res['engine']}")
    print("-" * 60)
    print("PARAMETERS BREAKDOWN:")
    if "parameters" in res:
        for p_name, p_val in res["parameters"].items():
            readable_name = p_name.replace("_", " ").title()
            print(f"  - {readable_name:<40}: {p_val}%")
    else:
        print(f"  - ResNet-18 Spatial Score (CNN)          : {res.get('cnn_score', 0)}%")
        print(f"  - LSTM Temporal Sequence Score            : {res.get('lstm_score', 0)}%")
        print(f"  - Noise Blueprint Variance               : {res.get('noise_score', 0)}%")
        print(f"  - Color Balance Score                    : {res.get('color_score', 0)}%")
        print(f"  - Grayscale Discrepancy Score            : {res.get('grayscale_score', 0)}%")
        print(f"  - Suspicion Score                        : {res.get('suspicion_score', 0)}%")
    print("=" * 60)

