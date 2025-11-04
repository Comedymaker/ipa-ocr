from __future__ import annotations

import torch
from torch import nn, Tensor


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for sequence-based OCR."""

    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x384 -> 64x192
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x192 -> 32x96
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 32x96 -> 16x96
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 16x96 -> 8x96
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # (B, 512, 1, W)
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, images: Tensor) -> Tensor:
        # images: (B, 1, H, W)
        features = self.cnn(images)  # (B, C, 1, T)
        features = features.squeeze(2)  # (B, C, T)
        features = features.permute(2, 0, 1).contiguous()  # (T, B, C)
        sequence, _ = self.rnn(features)  # (T, B, 2*hidden)
        logits = self.classifier(sequence)  # (T, B, vocab)
        return logits.log_softmax(dim=-1)
