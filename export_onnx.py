"""Export the ResNet18-LSTM deepfake detector to high-performance ONNX format."""
import os
import argparse
from pathlib import Path
import torch
from model import ResNetLSTMDetector
from utils import MODEL_PATH, MODELS_DIR, ensure_directories

os.environ["PYTHONIOENCODING"] = "utf-8"
ONNX_MODEL_PATH = MODELS_DIR / "deepfake_model.onnx"

def export_model_to_onnx(output_path=None, force=False):
    """Export trained or pretrained detector weights to ONNX format."""
    ensure_directories()
    out_file = Path(output_path) if output_path else ONNX_MODEL_PATH
    if out_file.exists() and not force:
        return out_file

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = ResNetLSTMDetector(pretrained=False, **checkpoint.get("model_config", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Pretrained ResNet-18 backbone + initialized LSTM classifier
        model = ResNetLSTMDetector(pretrained=True, hidden_size=256, lstm_layers=1, dropout=0.25)
    
    model.eval()
    sample = torch.randn(1, 1, 3, 224, 224)
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(out_file),
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={"frames": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    return out_file

def ensure_onnx_model(output_path=None, force=False):
    """Ensure ONNX model file exists; generate if missing."""
    out_file = Path(output_path) if output_path else ONNX_MODEL_PATH
    if not out_file.exists() or force:
        return export_model_to_onnx(out_file, force=force)
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Export deepfake detector to ONNX")
    parser.add_argument("--output", default=str(ONNX_MODEL_PATH), help="Target ONNX filepath")
    parser.add_argument("--force", action="store_true", help="Force re-export if file exists")
    args = parser.parse_args()
    res = export_model_to_onnx(args.output, force=args.force)
    print(f"ONNX model successfully saved to: {res}")

if __name__ == "__main__":
    main()
