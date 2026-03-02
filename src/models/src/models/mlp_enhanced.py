import torch
import torch.nn as nn


class EnhancedMLP(nn.Module):
    """增強版 MLP"""

    def __init__(self, num_classes=10):
        super(EnhancedMLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 自動展平
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def create_enhanced_mlp(num_classes=10):
    return EnhancedMLP(num_classes=num_classes)