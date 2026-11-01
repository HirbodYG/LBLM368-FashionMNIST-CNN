"""
Model tanımları
---------------
- FashionCNN : Projenin ana modeli (4 evrişim katmanı, BatchNorm, Dropout)
- SimpleMLP  : Karşılaştırma için tam bağlı (yoğun) temel model
"""
import torch.nn as nn


class FashionCNN(nn.Module):
    """
    Giriş: 1x28x28 gri tonlamalı görüntü
    Blok-1: Conv(32,3x3) -> BN -> ReLU -> Conv(32,3x3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)   => 32x14x14
    Blok-2: Conv(64,3x3) -> BN -> ReLU -> Conv(64,3x3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)   => 64x7x7
    Sınıflandırıcı: Flatten(3136) -> FC(256) -> BN -> ReLU -> Dropout(0.5) -> FC(10)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(0.25),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SimpleMLP(nn.Module):
    """Temel model: 784 -> 256 -> 128 -> 10 (ReLU, Dropout 0.2)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
