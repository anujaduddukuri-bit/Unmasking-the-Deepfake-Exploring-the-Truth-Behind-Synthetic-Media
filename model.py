"""ResNet feature extractor followed by an LSTM sequence classifier."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetLSTMDetector(nn.Module):
    def __init__(self, hidden_size=256, lstm_layers=1, dropout=0.25, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.lstm = nn.LSTM(self.feature_dim, hidden_size, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, 1))

    def extract_features(self, frames):
        """frames is [B, T, C, H, W]; returns [B, T, 512]."""
        batch, sequence = frames.shape[:2]
        values = self.feature_extractor(frames.reshape(batch * sequence, *frames.shape[2:]))
        return values.flatten(1).reshape(batch, sequence, -1)

    def forward(self, frames):
        features = self.extract_features(frames)
        sequence_output, _ = self.lstm(features)
        return self.classifier(sequence_output[:, -1]).squeeze(1)

    @torch.no_grad()
    def score_components(self, frames):
        features = self.extract_features(frames)
        output, _ = self.lstm(features)
        logits = self.classifier(output[:, -1]).squeeze(1)
        probability = torch.sigmoid(logits)
        # An interpretable model component signal, not an independent classifier.
        cnn_signal = torch.sigmoid(features.abs().mean(dim=(1, 2)) - 0.5)
        lstm_signal = torch.sigmoid(output[:, -1].abs().mean(dim=1) - 0.5)
        return probability, cnn_signal, lstm_signal
