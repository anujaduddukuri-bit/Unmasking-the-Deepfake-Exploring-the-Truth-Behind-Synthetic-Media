import base64
import json
import logging
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from inference import predict_image, predict_video
from utils import UPLOADS_DIR, VIDEO_FRAMES_DIR, allowed_file, allowed_video, ensure_directories

ensure_directories()

app = Flask(__name__)
app.config.update(UPLOAD_FOLDER=str(UPLOADS_DIR))
logging.basicConfig(level=logging.INFO)


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
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".bmp": "image/bmp"}.get(ext, "image/png")


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

@app.get("/video-results/<job_id>")
def video_results(job_id):
    report_file = VIDEO_FRAMES_DIR / job_id / "report.json"
    if not report_file.exists():
        abort(404, description="Video analysis job not found.")
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return render_template("video_results.html", result=data, job_id=job_id)

@app.get("/video-results/<job_id>/data")
def video_results_data(job_id):
    report_file = VIDEO_FRAMES_DIR / job_id / "report.json"
    if not report_file.exists():
        return jsonify(error="Job not found"), 404
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

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

        result = predict_image(path, demonstration_mode=True)

        # Embed all 3 images as base64 data URIs directly in JSON
        # This avoids separate HTTP requests and works on Vercel's read-only filesystem
        result["original_image"]  = _b64(path, _mime(path))
        result["heatmap_image"]   = _b64(result.get("heatmap_path", ""))
        result["blueprint_image"] = _b64(result.get("blueprint_path", ""))

        result.pop("heatmap_path", None)
        result.pop("blueprint_path", None)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify(error="Model weights not found."), 503
    except (UnidentifiedImageError, ValueError):
        return jsonify(error="That file is not a valid readable image."), 400
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify(error="Analysis could not be completed."), 500

@app.post("/analyze-video")
def analyze_video():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Please choose a video to analyze."), 400
    if not allowed_video(file.filename):
        return jsonify(error="Unsupported video type. Use MP4, AVI, MOV, MKV, or WEBM."), 400

    job_id = uuid4().hex
    filename = f"{job_id}_{secure_filename(file.filename)}"
    path = UPLOADS_DIR / filename
    try:
        file.save(path)
        result = predict_video(path, job_id, demonstration_mode=True)

        # Embed each frame's images as base64 data URIs
        for frame in result["frames"]:
            for field in ("original_path", "heatmap_path", "blueprint_path"):
                if field in frame:
                    img_key = field.replace("_path", "_image")
                    frame[img_key] = _b64(frame[field])
                    del frame[field]

        # Save report so /video-results/<job_id> page works
        report_file = VIDEO_FRAMES_DIR / job_id / "report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        result["redirect_url"] = f"/video-results/{job_id}"
        return jsonify(result)
    except (ValueError, UnidentifiedImageError):
        return jsonify(error="That file does not contain readable video frames."), 400
    except Exception:
        app.logger.exception("Video analysis failed")
        return jsonify(error="Video analysis could not be completed."), 500

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
