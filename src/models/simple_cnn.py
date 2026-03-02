"""
簡單 CNN 模型
2-3 層卷積網絡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """簡單卷積神經網絡"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 卷積層 1: 3 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )

        # 卷積層 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )

        # 卷積層 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )

        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 展平
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_simple_cnn(num_classes=10):
    """創建簡單 CNN 模型"""
    return SimpleCNN(num_classes=num_classes)


if __name__ == '__main__':
    # 測試模型
    model = create_simple_cnn(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"輸入形狀：{x.shape}")
    print(f"輸出形狀：{y.shape}")
    print(f"模型參數：{sum(p.numel() for p in model.parameters()):,}")