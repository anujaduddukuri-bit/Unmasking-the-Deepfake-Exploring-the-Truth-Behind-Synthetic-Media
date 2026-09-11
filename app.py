import json
import logging
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, url_for, abort
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from inference import predict_image, predict_video
from utils import UPLOADS_DIR, VIDEO_FRAMES_DIR, HEATMAPS_DIR, allowed_file, allowed_video, ensure_directories

ensure_directories()


app = Flask(__name__)
app.config.update(UPLOAD_FOLDER=str(UPLOADS_DIR))
logging.basicConfig(level=logging.INFO)

@app.get("/")
def welcome():
    """Welcome Landing Page."""
    return render_template("welcome.html")

@app.get("/image")
def image_lab():
    """Single Image Forensics Lab."""
    return render_template("index.html")

@app.get("/index")
def index():
    return render_template("index.html")

@app.get("/video")
def video():
    return render_template("video.html")

@app.get("/video-results/<job_id>")
def video_results(job_id):
    """Dedicated standalone video forensics report page."""
    report_file = VIDEO_FRAMES_DIR / job_id / "report.json"
    if not report_file.exists():
        abort(404, description="Video analysis job not found.")
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for frame in data.get("frames", []):
        for prefix in ("original", "blueprint", "heatmap"):
            img_key = f"{prefix}_image"
            path_key = f"{prefix}_path"
            if img_key not in frame:
                if path_key in frame:
                    fname = Path(frame[path_key]).name
                else:
                    if prefix == "original":
                        fname = f"frame_{frame['number']:03d}.jpg"
                    elif prefix == "blueprint":
                        fname = f"frame_{frame['number']:03d}_noise_blueprint.png"
                    else:
                        fname = f"frame_{frame['number']:03d}_heatmap.png"
                frame[img_key] = url_for("video_file", job_id=job_id, filename=fname)
    return render_template("video_results.html", result=data, job_id=job_id)

@app.get("/video-results/<job_id>/data")
def video_results_data(job_id):
    """API endpoint to retrieve structured report for a video job."""
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
        result["original_image"] = url_for("static", filename=f"uploads/{filename}")
        result["heatmap_image"] = url_for("generated", filename=Path(result["heatmap_path"]).name)
        result["blueprint_image"] = url_for("generated", filename=Path(result["blueprint_path"]).name)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify(error="Model weights not found. Train or export the model first."), 503
    except (UnidentifiedImageError, ValueError):
        return jsonify(error="That file is not a valid readable image."), 400
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify(error="Analysis could not be completed. Check the model and image, then try again."), 500

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
        
        # Populate web URLs for every separate frame
        for frame in result["frames"]:
            for field in ("original_path", "heatmap_path", "blueprint_path"):
                if field in frame:
                    frame[field.replace("_path", "_image")] = url_for(
                        "video_file", job_id=job_id, filename=Path(frame[field]).name
                    )
                    del frame[field]
                    
        # Update saved report.json with the generated web URLs
        report_file = VIDEO_FRAMES_DIR / job_id / "report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            
        result["redirect_url"] = url_for("video_results", job_id=job_id)
        return jsonify(result)
    except (ValueError, UnidentifiedImageError):
        return jsonify(error="That file does not contain readable video frames."), 400
    except Exception:
        app.logger.exception("Video analysis failed")
        return jsonify(error="Video analysis could not be completed."), 500

@app.get("/generated/<filename>")
def generated(filename):
    from flask import send_from_directory
    return send_from_directory(HEATMAPS_DIR, filename)

@app.get("/video-frames/<job_id>/<filename>")
def video_file(job_id, filename):
    from flask import send_from_directory
    return send_from_directory(VIDEO_FRAMES_DIR / job_id, filename)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
