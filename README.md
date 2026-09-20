# 🔍 Unmasking the Deepfake: Exploring the Truth Behind Synthetic Media

> **A Hybrid AI-Powered Digital Forensics System for Detecting Deepfakes in Images and Videos**

## 📌 Overview

**Unmasking the Deepfake: Exploring the Truth Behind Synthetic Media** is an AI-powered deepfake detection and digital forensics system designed to analyze images and videos and determine whether the provided media is **REAL** or **FAKE**.

The system combines a **ResNet-18 Convolutional Neural Network (CNN)** for visual feature extraction with an **LSTM sequence model** for temporal analysis. The trained detection pipeline is exported to **ONNX Runtime** to provide efficient inference.

In addition to neural-network-based detection, the system performs multiple forensic analyses, including **noise residual analysis, color analysis, grayscale analysis, suspicious-region heatmaps, and video temporal forensics**.

The goal is not only to provide a final prediction but also to present **visual forensic evidence** that helps users understand why a piece of media may contain synthetic or manipulated content.

---

## 🎯 Objectives

The main objectives of the project are:

* Detect AI-generated and manipulated images.
* Detect deepfake and manipulated videos.
* Classify media as **REAL** or **FAKE**.
* Extract visual features using a CNN-based model.
* Analyze sequential video frames using an LSTM.
* Identify suspicious regions using forensic heatmaps.
* Analyze high-frequency noise and image residuals.
* Perform color and grayscale forensic analysis.
* Analyze temporal inconsistencies between video frames.
* Provide visual forensic evidence alongside the prediction.
* Provide a dedicated interface for image and video analysis.

---

## 🧠 System Architecture

The project follows a hybrid deep-learning and digital-forensics pipeline.

```text
                    ┌─────────────────────┐
                    │   User Uploads      │
                    │ Image / Video       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Preprocessing    │
                    │ Resize / Normalize  │
                    │ Frame Extraction    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌──────────────────┐
        │ ResNet-18 CNN   │        │ Forensic Analysis│
        │ Feature         │        │                  │
        │ Extraction      │        │ Noise / Color   │
        └────────┬────────┘        │ Grayscale       │
                 │                 │ Heatmap         │
                 ▼                 └────────┬─────────┘
        ┌─────────────────┐                 │
        │ 512-D Visual    │                 │
        │ Features        │                 │
        └────────┬────────┘                 │
                 │                           │
                 ▼                           │
        ┌─────────────────┐                  │
        │      LSTM       │                  │
        │ Sequence Model  │                  │
        └────────┬────────┘                  │
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Final Analysis    │
                    │                     │
                    │ REAL / FAKE         │
                    │ Scores + Evidence   │
                    └─────────────────────┘
```

---

## 🤖 AI Detection Model

The core detection pipeline uses a hybrid **CNN + LSTM architecture**.

### ResNet-18 CNN

ResNet-18 is used to extract visual features from images and video frames.

* Extracts **512-dimensional visual features**.
* Captures spatial patterns and visual inconsistencies.
* Processes individual images or video frames.

### LSTM Sequence Model

For video analysis, extracted frame features are processed sequentially using an LSTM.

* Hidden size: **256**
* Captures temporal relationships between frames.
* Helps identify inconsistencies and unnatural changes across consecutive frames.

### Classification Head

The sequence model uses the following classification structure:

```text
LSTM
  ↓
Linear(256 → 128)
  ↓
ReLU
  ↓
Dropout(0.25)
  ↓
Linear(128 → 1)
  ↓
Sigmoid
  ↓
REAL / FAKE
```

---

## ⚡ ONNX Runtime

The detection pipeline can be exported to **ONNX format** for optimized inference.

The project includes:

```text
models/deepfake_model.onnx
```

ONNX Runtime provides a lightweight inference engine that can be used for efficient model execution across supported environments.

---

## 🔬 Digital Forensics Modules

The project does more than simply classify an image or video. It also generates multiple forensic signals.

### 1. Noise Forensics

The noise-analysis module examines high-frequency image information to identify possible synthesis or manipulation artifacts.

It analyzes:

* Median residuals
* Gaussian residuals
* Laplacian texture
* Local noise variance
* High-frequency discrepancies

The system generates a **Noise Blueprint** and **Noise Overlay** to visually represent suspicious patterns.

---

### 2. Color Forensics

The color-analysis module examines:

* RGB channel balance
* HSV saturation
* Exposure variations
* LAB color residuals

These measurements can reveal unusual color distributions and inconsistencies.

---

### 3. Grayscale Forensics

Grayscale analysis focuses on luminance and edge information.

It examines:

* Multi-scale luminance inconsistencies
* Laplacian edge responses
* Texture variations

---

### 4. Suspicious Region Heatmap

The heatmap module identifies regions that show stronger forensic signals.

The system generates:

* Heatmap visualization
* Suspicious-region mask
* Evidence overlay

These visualizations help users locate areas that contribute to the forensic analysis.

---

## 🎥 Video Temporal Forensics

Video analysis uses multiple frames rather than relying on a single frame.

The system:

* Samples frames across the video timeline.
* Supports up to **30 FPS**.
* Processes up to **30 separate frames**.
* Analyzes frame-to-frame changes.
* Measures temporal flickering.
* Examines optical-flow luminance changes.
* Measures noise-variance jitter.
* Generates a temporal forensic timeline.

The video results page presents the extracted frames individually along with their forensic information.

---

## 🖥️ Application Features

### Image Forensics Lab

The image analysis interface allows users to:

1. Upload an image.
2. Process the image through the detection pipeline.
3. Obtain a REAL/FAKE prediction.
4. View forensic scores.
5. Examine noise analysis.
6. View color and grayscale analysis.
7. Inspect suspicious-region heatmaps.
8. Review visual evidence.

### Video Forensics Lab

The video interface allows users to:

1. Upload a video.
2. Extract representative frames.
3. Analyze frames using the detection pipeline.
4. Perform temporal forensic analysis.
5. View frame-level predictions.
6. Examine forensic visualizations.
7. Review the overall video prediction.

### Dedicated Video Results

Video analysis produces a dedicated results page containing:

* Extracted video frames
* Original frame previews
* Noise blueprint
* Noise overlay
* Heatmap
* Suspicious-region information
* Frame-level scores
* Temporal analysis
* Interactive timeline
* Frame inspection interface

---

## 🛠️ Technology Stack

| Technology              | Purpose                    |
| ----------------------- | -------------------------- |
| **Python**              | Core programming language  |
| **Flask**               | Web application framework  |
| **ONNX Runtime**        | Model inference            |
| **OpenCV**              | Image/video processing     |
| **NumPy**               | Numerical computation      |
| **Pillow**              | Image processing           |
| **ResNet-18**           | Visual feature extraction  |
| **LSTM**                | Temporal sequence analysis |
| **HTML/CSS/JavaScript** | Web interface              |
| **Gunicorn**            | Production WSGI server     |

The project's current dependency file includes Flask, OpenCV, NumPy, Pillow, ONNX Runtime, and Gunicorn.

---

## 📁 Project Structure

```text
Unmasking-the-Deepfake-Exploring-the-Truth-Behind-Synthetic-Media/
│
├── models/
│   └── deepfake_model.onnx
│
├── results/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── welcome.html
│   ├── index.html
│   ├── video.html
│   └── about.html
│
├── uploads/
│
├── app.py
├── model.py
├── inference.py
├── preprocessing.py
├── train.py
├── export_onnx.py
│
├── color_analysis.py
├── grayscale_analysis.py
├── heatmap.py
├── noise_analysis.py
├── video_forensics.py
├── utils.py
│
├── requirements.txt
├── vercel.json
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/anujaduddukuri-bit/Unmasking-the-Deepfake-Exploring-the-Truth-Behind-Synthetic-Media.git
```

### 2. Open the project directory

```bash
cd Unmasking-the-Deepfake-Exploring-the-Truth-Behind-Synthetic-Media
```

### 3. Create a virtual environment

Windows:

```powershell
py -m venv venv
```

### 4. Activate the virtual environment

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application:

```powershell
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

### Main Pages

| Page            | URL                           |
| --------------- | ----------------------------- |
| Home            | `http://127.0.0.1:5000/`      |
| Image Forensics | `http://127.0.0.1:5000/image` |
| Video Forensics | `http://127.0.0.1:5000/video` |
| About           | `http://127.0.0.1:5000/about` |

---

## 🧪 Command-Line Inference

### Image Analysis

```powershell
python inference.py path/to/image.jpg
```

Example:

```powershell
python inference.py fake/DF.jpeg
```

### Video Analysis

```powershell
python inference.py "path/to/video.mp4"
```

Example:

```powershell
python inference.py "video DF.mp4"
```

---

## 🏋️ Model Training

If training or fine-tuning is required, the training script can be executed using:

```powershell
python train.py --epochs 10 --batch-size 8
```

After training, the model can be exported to ONNX using:

```powershell
python export_onnx.py
```

---

## 🔄 Detection Workflow

```text
Upload Media
     ↓
Validate Input
     ↓
Preprocess Image / Video
     ↓
Extract Frames
     ↓
ResNet-18 Feature Extraction
     ↓
LSTM Temporal Analysis
     ↓
ONNX Runtime Inference
     ↓
Forensic Analysis
     ├── Noise Analysis
     ├── Color Analysis
     ├── Grayscale Analysis
     ├── Heatmap Analysis
     └── Temporal Analysis
     ↓
Generate Evidence
     ↓
REAL / FAKE Prediction
     ↓
Display Forensic Report
```

---

## 📊 Output

The system provides a combination of:

* Final media classification
* Detection scores
* Frame-level analysis
* Noise visualizations
* Color analysis
* Grayscale analysis
* Suspicious-region heatmaps
* Temporal inconsistencies
* Forensic evidence overlays

This makes the system useful for **visual inspection and forensic analysis**, rather than relying only on a single classification label.

---

## 🎓 Academic Project

**Project Title:**
**Unmasking the Deepfake: Exploring the Truth Behind Synthetic Media**

**Project Domain:**
Artificial Intelligence & Machine Learning

**Key Areas:**

* Deep Learning
* Computer Vision
* Digital Image Forensics
* Video Forensics
* Synthetic Media Detection
* CNN
* LSTM
* ONNX
* Flask

---

## 🚀 Future Enhancements

Possible future improvements include:

* Support for additional deepfake architectures.
* Improved model accuracy through larger and more diverse datasets.
* Audio-based deepfake detection.
* Multimodal audio-video analysis.
* Face-level manipulation localization.
* Real-time video stream analysis.
* Explainable AI techniques for model decisions.
* Advanced transformer-based detection models.
* Improved deployment scalability.
* Automated forensic report generation.

---

## ⚠️ Limitations

Deepfake detection is an evolving research problem. Detection performance can vary depending on:

* Image and video quality.
* Compression artifacts.
* Unseen manipulation techniques.
* Lighting conditions.
* Resolution.
* Video frame rate.
* Types of synthetic media used.

Therefore, the system's output should be considered as **forensic analysis evidence rather than absolute proof of authenticity**.

---

## 🔐 Privacy

Uploaded media may contain sensitive visual information. Users should avoid uploading confidential or personally sensitive media to deployments that are not under their control.

For production deployment, appropriate storage, access-control, retention, and privacy policies should be implemented.

---

## 📜 License

This project is intended for **academic, educational, and research purposes**.

---

## 👩‍💻 Author

**Anuja Duddukuri**

AI/ML Engineer | Full Stack Developer | AI Enthusiast

GitHub:
https://github.com/anujaduddukuri-bit

---

## ⭐ Acknowledgement

This project demonstrates the application of **Artificial Intelligence, Deep Learning, Computer Vision, and Digital Forensics** to the growing challenge of synthetic media and deepfake detection.

If you find this project useful for learning or research, consider giving the repository a ⭐.
