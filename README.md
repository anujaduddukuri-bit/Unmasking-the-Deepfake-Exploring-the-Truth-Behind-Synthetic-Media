# UNMASKING THE DEEPFAKE: HYBRID FORENSIC & ONNX SEQUENCE DETECTION

## Objective and Architecture

This forensic detection system analyzes images and videos as **REAL MEDIA** or **DEEPFAKE** using a hybrid deepfake-detection pipeline combining deep neural networks with multi-scale digital forensics:

1. **Learned Sequence Model (`model.py` & `models/deepfake_model.onnx`)**:
   - **ResNet-18 CNN** extracts 512-dimensional visual features from each image or video frame.
   - **LSTM sequence classifier** (hidden size 256) processes sequential frame feature vectors.
   - **Classifier Head**: `Linear(256→128) → ReLU → Dropout(0.25) → Linear(128→1) → Sigmoid`.
   - **High-Performance ONNX Runtime Engine**: The entire ResNet-LSTM pipeline is exported to ONNX (`models/deepfake_model.onnx`) for optimized, cross-platform inference.
2. **Sand-Pour Noise Blueprint & Overlay Forensics (`noise_analysis.py`)**:
   - **Sand-Pour Noise Blueprint**: Extracts multi-scale high-frequency noise residuals (median difference, Gaussian residual, Laplacian texture) and renders an architectural blueprint on a midnight Prussian navy substrate (`#030a16`). Granular sand stippling reveals generative synthesis artifacts and tampering boundaries.
   - **Pour-Sand Noise Overlay**: Simulates physical sand grains poured over the original image or frame, settling densely where local noise variance and high-frequency discrepancies concentrate.
3. **Multi-Perspective Forensics Suite**:
   - **Colour Forensics (`color_analysis.py`)**: RGB channel balance, HSV saturation/exposure, and LAB colour residuals.
   - **Grayscale Forensics (`grayscale_analysis.py`)**: Multi-scale luminance inconsistency and Laplacian edge responses.
   - **Suspicious Region Heatmap & Evidence Overlay (`heatmap.py`)**: OpenCV JET colormap and 92nd-percentile red evidence mask.
4. **Video Temporal Forensics (Max 30 FPS & 30 Separate Frames) (`preprocessing.py` & `video_forensics.py`)**:
   - Extracts up to 30 separate frames uniformly sampled across the video timeline, strictly capped at **30 FPS**.
   - Computes inter-frame temporal flickering, optical flow luminance shift, and noise variance jitter across adjacent frames.
   - Generates a frame-by-frame temporal forensic timeline.
5. **Dedicated Standalone Video Forensics Page (`/video-results/<job_id>`)**:
   - Video results are displayed on an independent, dedicated results page completely separate from the image analysis page.
   - Displays all 30 extracted frames separately in individual cards with full multi-view forensic thumbnails (Original, Sand Blueprint, Sand Overlay, Heatmap, Red Overlay) and detailed scores.
   - Interactive SVG temporal timeline chart and full-screen separate frame inspector modal with tab switching and chronological / suspicion sorting.

---

## Installation & Running

```powershell
# 1. Install dependencies
py -3.14 -m pip install -r requirements.txt

# 2. (Optional) Train or fine-tune detector on dataset/
py -3.14 train.py --epochs 10 --batch-size 8

# 3. Export model to ONNX format
py -3.14 export_onnx.py

# 4. Launch Flask web application
py -3.14 app.py
```

Open `http://127.0.0.1:5000` in your web browser:
- **Image Forensics Lab**: `http://127.0.0.1:5000/`
- **Video Frame Lab (Max 30 FPS)**: `http://127.0.0.1:5000/video`
- **Dedicated Video Forensics Report**: Automatically redirected to `http://127.0.0.1:5000/video-results/<job_id>`

---

## CLI Inference

```powershell
# Image Analysis
py -3.14 inference.py fake/DF.jpeg

# Video Analysis (Max 30 FPS, 30 Separate Frames)
py -3.14 inference.py "video DF.mp4"
```

