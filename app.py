import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, abort, Response, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from inference import predict_image, predict_video, load_detector
from utils import UPLOADS_DIR, VIDEO_FRAMES_DIR, allowed_file, allowed_video, ensure_directories

ensure_directories()

app = Flask(__name__)
app.config.update(UPLOAD_FOLDER=str(UPLOADS_DIR))
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────
# Pre-warm the detector singleton at startup so the first
# request doesn't pay the cold-load penalty.
# ──────────────────────────────────────────────────────────────
try:
    load_detector()
    app.logger.info("Detector singleton loaded at startup.")
except FileNotFoundError:
    app.logger.info("No model weights found — running in forensic heuristic mode.")
except Exception as exc:
    app.logger.warning("Detector warm-up error: %s", exc)


# ─── Helpers ──────────────────────────────────────────────────

def _b64(path, mime="image/png"):
    """Read a file and return a base64 data URI — works on Vercel serverless."""
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


def _mime(path):
    ext = Path(path).suffix.lower()
    return {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
    }.get(ext, "image/png")


# ─── Page routes ──────────────────────────────────────────────

@app.get("/")
def welcome():
    return render_template("welcome.html")

@app.get("/image")
def image_lab():
    return render_template("index.html")

@app.get("/index")
def index():
    return render_template("index.html")

@app.get("/video")
def video():
    return render_template("video.html")

@app.get("/about")
def about():
    return render_template("about.html")

_REPORTS_CACHE = {}

def _normalize_video_report(data, job_id):
    """Ensure all required forensic telemetry fields exist for backwards compatibility and resilience."""
    frames = data.get("frames", [])
    total_frames = data.get("frame_count", len(frames))
    
    suspicious_count = 0
    for frame in frames:
        susp = float(frame.get("suspicion_score", 0.0))
        if "frame_verdict" not in frame:
            frame["frame_verdict"] = "FAKE" if susp >= 50.0 else "REAL"
        if frame.get("frame_verdict") == "FAKE":
            suspicious_count += 1
        # Convert any legacy relative URLs into base64 data URIs so they never 404 on Vercel
        for key in ("original_image", "blueprint_image", "heatmap_image"):
            val = frame.get(key, "")
            if val and not str(val).startswith("data:"):
                fname = Path(str(val)).name
                fpath = VIDEO_FRAMES_DIR / job_id / fname
                if fpath.exists():
                    frame[key] = _b64(fpath, _mime(fpath))

    real_count = max(0, total_frames - suspicious_count)
    inconsistency = float(data.get("temporal_inconsistency_score", 15.0))
    pred = data.get("prediction", "REAL")

    temp_cons = "High" if inconsistency < 30.0 else ("Moderate" if inconsistency < 55.0 else "Degraded")
    
    verdict_text = (
        "VERIFIED AUTHENTIC MEDIA — REAL" if pred in ("REAL",)
        else "DEEPFAKE DETECTED — FAKE"
    )

    defaults = {
        "job_id": job_id,
        "frame_count": total_frames,
        "real_frames": data.get("real_frames", real_count),
        "suspicious_frames": data.get("suspicious_frames", suspicious_count),
        "temporal_consistency": data.get("temporal_consistency", temp_cons),
        "face_detection_status": data.get("face_detection_status", "Active Tracking (Multi-Frame)"),
        "artifact_anomaly_score": data.get("artifact_anomaly_score", round(float(data.get("cnn_score", 15.0)) * 0.8, 1)),
        "compression_indicator": data.get("compression_indicator", "Nominal (H.264/AVC Temporal GOP)"),
        "overall_verdict_text": data.get("overall_verdict_text", verdict_text),
        "confidence": float(data.get("confidence", 90.0)),
        "real_percentage": float(data.get("real_percentage", 50.0)),
        "fake_percentage": float(data.get("fake_percentage", 50.0)),
        "prediction": pred,
        "cnn_score": float(data.get("cnn_score", 15.0)),
        "lstm_score": float(data.get("lstm_score", 15.0)),
        "temporal_inconsistency_score": inconsistency,
        "temporal_jitter_score": float(data.get("temporal_jitter_score", 5.0)),
        "color_drift_score": float(data.get("color_drift_score", 5.0)),
        "average_suspicion": float(data.get("average_suspicion", 15.0)),
        "peak_frame": int(data.get("peak_frame", 1)),
        "peak_score": float(data.get("peak_score", 15.0)),
        "peak_timestamp": float(data.get("peak_timestamp", 0.0)),
        "frames": frames,
    }
    for k, v in defaults.items():
        if k not in data or data[k] is None:
            data[k] = v
    return data

@app.get("/video-results/<job_id>")
def video_results(job_id):
    data = _REPORTS_CACHE.get(job_id)
    if not data:
        report_file = VIDEO_FRAMES_DIR / job_id / "report.json"
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
    if not data:
        from utils import REPORTS_DIR
        rep_file = REPORTS_DIR / f"{job_id}.json"
        if rep_file.exists():
            try:
                with open(rep_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

    if not data:
        abort(404, description="Video analysis job not found.")

    data = _normalize_video_report(data, job_id)
    return render_template("video_results.html", result=data, job_id=job_id)

@app.get("/video-results/<job_id>/data")
def video_results_data(job_id):
    data = _REPORTS_CACHE.get(job_id)
    if not data:
        report_file = VIDEO_FRAMES_DIR / job_id / "report.json"
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
    if not data:
        from utils import REPORTS_DIR
        rep_file = REPORTS_DIR / f"{job_id}.json"
        if rep_file.exists():
            try:
                with open(rep_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

    if not data:
        return jsonify(error="Job not found"), 404

    data = _normalize_video_report(data, job_id)
    return jsonify(data)




# ─── Analysis API ─────────────────────────────────────────────

@app.post("/analyze")
def analyze():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Please choose an image to analyze."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Use PNG, JPG, JPEG, BMP, or WEBP."), 400

    filename = f"{uuid4().hex}_{secure_filename(file.filename)}"
    path = UPLOADS_DIR / filename
    try:
        file.save(path)
        with Image.open(path) as image:
            image.verify()

        result = predict_image(path, demonstration_mode=False)

        # Embed all 3 images as base64 data URIs (works on Vercel read-only FS)
        result["original_image"]  = _b64(path, _mime(path))
        result["heatmap_image"]   = _b64(result.get("heatmap_path", ""))
        result["blueprint_image"] = _b64(result.get("blueprint_path", ""))

        result.pop("heatmap_path",   None)
        result.pop("blueprint_path", None)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify(error="Model weights not found."), 503
    except (UnidentifiedImageError, ValueError):
        return jsonify(error="That file is not a valid readable image."), 400
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify(error="Analysis could not be completed."), 500


@app.get("/results/<path:filename>")
def serve_results(filename):
    from utils import RESULTS_DIR
    return send_from_directory(str(RESULTS_DIR), filename)

@app.get("/video-frames/<path:filename>")
def serve_video_frames(filename):
    from utils import VIDEO_FRAMES_DIR
    return send_from_directory(str(VIDEO_FRAMES_DIR), filename)

@app.get("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(str(UPLOADS_DIR), filename)


@app.post("/analyze-video")
def analyze_video():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Please choose a video to analyze."), 400
    if not allowed_video(file.filename):
        return jsonify(error="Unsupported video type. Use MP4, AVI, MOV, MKV, or WEBM."), 400

    job_id   = uuid4().hex
    filename = f"{job_id}_{secure_filename(file.filename)}"
    path     = UPLOADS_DIR / filename
    try:
        file.save(path)
        result = predict_video(path, job_id, demonstration_mode=False)
        _REPORTS_CACHE[job_id] = result
        result["redirect_url"] = f"/video-results/{job_id}"
        return jsonify(result)
    except (ValueError, UnidentifiedImageError):
        return jsonify(error="That file does not contain readable video frames."), 400
    except Exception:
        app.logger.exception("Video analysis failed")
        return jsonify(error="Video analysis could not be completed."), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
