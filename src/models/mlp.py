import torch
import torch.nn as nn

class MLP(nn.Module):
    """多層感知機分類器"""

    def __init__(self, input_size=32*32*3, num_classes=10, hidden_sizes=[512, 256, 128]):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # 隱藏層
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        # 輸出層
        layers.append(nn.Linear(prev_size, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # 關鍵：展平圖像
        # [batch, 3, 32, 32] -> [batch, 3072]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_mlp(num_classes=10):
    """創建 MLP 模型"""
    return MLP(input_size=32*32*3, num_classes=num_classes)