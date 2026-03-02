"""
使用 CIFAR-10 作為測試數據集
快速驗證模型代碼，後續可切換到 Places365
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 70)
print(" " * 20 + "CIFAR-10 數據集準備")
print("=" * 70)

# CIFAR-10 類別
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 數據轉換
print("\n📥 正在下載/加載 CIFAR-10...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# 下載數據集
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

print(f"\n✅ CIFAR-10 加載成功！")
print(f"   訓練集: {len(train_dataset)} 張圖像")
print(f"   測試集: {len(test_dataset)} 張圖像")
print(f"   圖像尺寸: 32x32 RGB")
print(f"   類別數: {len(CLASSES)}")
print(f"   類別: {', '.join(CLASSES)}")

# 創建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print(f"\n📊 DataLoader 統計：")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches: {len(test_loader)}")

# 可視化樣本
print("\n📊 可視化樣本圖像...")


def imshow(img, title):
    """顯示圖像"""
    img = img / 2 + 0.5  # 反歸一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')


# 獲取一個批次的數據
images, labels = next(iter(train_loader))

# 創建圖表
fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    if i < len(images):
        img = images[i]
        label = CLASSES[labels[i].item()]

        # 反歸一化
        img = img / 2 + 0.5
        npimg = img.numpy()

        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(label, fontsize=8)
        ax.axis('off')

plt.suptitle("CIFAR-10 樣本圖像", fontsize=12, fontweight='bold')
plt.tight_layout()

# 保存圖像
save_path = './notebooks/cifar10_samples.png'
os.makedirs('./notebooks', exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✅ 樣本圖像已保存: {save_path}")

plt.show()

# 測試 GPU
print("\n" + "=" * 70)
print(" " * 20 + "GPU 測試")
print("=" * 70)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"   顯存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2:.0f} MB")
else:
    device = torch.device('cpu')
    print("⚠️  使用 CPU 模式")

# 測試數據轉移到 GPU
images_gpu = images.to(device)
print(f"\n✅ 數據成功轉移到 {device}")

print("\n" + "=" * 70)
print(" " * 15 + "🎉 CIFAR-10 準備完成！")
print("=" * 70)
print("\n下一步：")
print("  1. 開始編寫第一個 CNN 模型")
print("  2. 實現訓練循環")
print("  3. 測試和評估")
print("\n提示：")
print("  - CIFAR-10 很小，適合快速測試代碼")
print("  - 代碼結構完成後，可以切換到 Places365")
print()