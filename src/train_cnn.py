"""
訓練腳本
支持 MLP 和 CNN 模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
import json
from pathlib import Path

# 導入模型
from models.mlp import create_mlp
from models.simple_cnn import create_simple_cnn


def get_data_loaders(batch_size=64, num_workers=0):
    """獲取 CIFAR-10 數據加載器"""

    # 數據增強（訓練集）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # 測試集（不做增強）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # 加載數據集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向傳播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 統計
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """評估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train(model_name='cnn', epochs=20, batch_size=64, lr=0.001):
    """主訓練函數"""

    print("=" * 70)
    print(" " * 20 + f"訓練 {model_name.upper()} 模型")
    print("=" * 70)

    # ========== 添加調試信息 ==========
    print("\n🔍 GPU 診斷：")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.version.cuda: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
    if torch.cuda.is_available():
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"  torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  CUDA 不可用，將使用 CPU")
    # ==================================

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 設備：{device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 創建模型
    print(f"\n📦 創建模型...")
    if model_name == 'mlp':
        model = create_mlp(num_classes=10)
    elif model_name == 'cnn':
        model = create_simple_cnn(num_classes=10)
    else:
        raise ValueError(f"未知模型：{model_name}")

    model = model.to(device)

    # 統計參數
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   模型參數：{num_params:,}")

    # 數據加載器
    print(f"\n📊 加載數據...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 訓練記錄
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # 開始訓練
    print(f"\n🚀 開始訓練...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print("-" * 70)

    best_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        # 訓練
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 評估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 學習率調整
        scheduler.step()

        # 記錄
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存模型
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f'./models/best_{model_name}.pth')

        # 打印進度
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

    print("-" * 70)
    print(f"\n✅ 訓練完成！")
    print(f"   最佳測試準確率：{best_acc:.2f}%")
    print(f"   模型保存：./models/best_{model_name}.pth")


    # 訓練完成後保存 history
    print("\n💾 保存訓練歷史...")

    # 保存為 JSON
    os.makedirs('./experiments/results', exist_ok=True)
    with open(f'./experiments/results/{model_name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 保存為 torch
    torch.save(history, f'./experiments/results/{model_name}_history.pth')

    print(f"✅ 歷史記錄已保存：./experiments/results/{model_name}_history.json")

    return history, model


#if __name__ == '__main__':
    # 訓練 CNN
    # history, model = train(model_name='cnn', epochs=20, batch_size=64, lr=0.001)


if __name__ == '__main__':
    # 訓練 MLP
    history, model = train(model_name='mlp', epochs=50, batch_size=128, lr=0.01)