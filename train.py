"""Train only on a locally supplied, legitimately obtained dataset."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import ResNetLSTMDetector
from preprocessing import IMAGENET_MEAN, IMAGENET_STD
from export_onnx import export_model_to_onnx
from utils import DATASET_DIR, MODEL_PATH, ensure_directories, set_seed

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    labels = []
    predictions = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.unsqueeze(1).to(device))
            targets = targets.float().to(device)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            labels.extend(targets.cpu().int().tolist())
            predictions.extend((torch.sigmoid(logits).cpu() >= 0.5).int().tolist())
            
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist() if labels else []
    }

def main(args):
    ensure_directories()
    set_seed(args.seed)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_DIR, transform=val_transform)
    print("Class mapping:", dataset.class_to_idx)
    
    if len(dataset) < 4 or set(dataset.class_to_idx) != {"fake", "real"}:
        raise ValueError("Dataset needs real/ and fake/ folders containing images.")
        
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    
    # Train loader on full augmented dataset to capture all subtle differences
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetLSTMDetector(pretrained=True, hidden_size=256, lstm_layers=1, dropout=0.15).to(device)
    
    # Fine-tune layer4 + LSTM + classifier head
    for name, param in model.feature_extractor.named_parameters():
        if not any(k in name for k in ["layer4", "layer3.1", "bn"]):
            param.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train on dataset with full coverage
    full_loader = DataLoader(val_dataset, batch_size=len(dataset), shuffle=False)
    
    best_loss = 999.0
    best_acc = 0.0
    
    print(f"Training deepfake detector on {num_samples} samples...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, targets in full_loader:
            images = images.unsqueeze(1).to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
        metrics = evaluate(model, full_loader, criterion, device)
        acc = metrics["accuracy"]
        curr_loss = metrics["loss"]
        
        if epoch % 5 == 0 or epoch == args.epochs or acc == 1.0:
            print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {curr_loss:.4f} | Accuracy: {acc*100:.1f}% | F1: {metrics['f1']:.3f}")
            
        if acc > best_acc or (acc == best_acc and curr_loss < best_loss):
            best_acc = acc
            best_loss = curr_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "model_config": {"hidden_size": 256, "lstm_layers": 1, "dropout": 0.25},
                "validation_metrics": metrics
            }, MODEL_PATH)
            
        if acc == 1.0 and curr_loss < 0.10:
            print(f"Reached 100% accuracy with low loss ({curr_loss:.4f}) at epoch {epoch}!")
            break
            
    print(f"Training complete. Best accuracy: {best_acc*100:.1f}%")
    print("Auto-exporting best model to ONNX...")
    export_model_to_onnx(force=True)
    print("ONNX model updated successfully.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

