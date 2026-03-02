"""
混淆矩陣分析
分析模型在各個類別上的表現
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

# 添加 src 目錄到路徑（確保可以導入模型）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CIFAR-10 類別名稱
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def load_model(model_path, num_classes=10):
    """加載訓練好的 CNN 模型"""
    from models.simple_cnn import create_simple_cnn

    print(f"📦 加載模型：{model_path}")

    # 創建模型架構
    model = create_simple_cnn(num_classes=num_classes)

    # 加載權重
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        # CPU 模式加載
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    print(f"✅ 模型加載成功")

    return model

def get_predictions(model, test_loader, device):
    """獲取模型在測試集上的預測結果"""

    print(f"\n🔮 進行預測...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向傳播
            outputs = model(inputs)
            _, preds = outputs.max(1)

            # 收集結果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 進度顯示
            if (batch_idx + 1) % 20 == 0:
                print(f"   已處理 {batch_idx + 1}/{len(test_loader)} 批次...")

    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path='./experiments/results/confusion_matrix.png',
                          normalize=False):
    """
    繪製混淆矩陣

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        class_names: 類別名稱列表
        save_path: 保存路徑
        normalize: 是否歸一化（顯示百分比）
    """

    print("\n" + "=" * 70)
    print(" " * 20 + "混淆矩陣分析")
    print("=" * 70)

    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)

    # 創建圖表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) if normalize else plt.subplots(1, 1, figsize=(10, 8))

    if normalize:
        axes_list = axes
    else:
        axes_list = [axes]

    # ========== 圖 1：原始計數 ==========
    if not normalize:
        ax = axes
    else:
        ax = axes_list[0]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Confusion Matrix (Count)', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    # ========== 圖 2：歸一化（百分比）==========
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # 處理除零

        ax2 = axes_list[1]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Proportion'}, ax=ax2)

        ax2.set_xlabel('Predicted Label', fontsize=11)
        ax2.set_ylabel('True Label', fontsize=11)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 混淆矩陣已保存：{save_path}")

    # 顯示
    plt.show()

    return cm

def print_analysis(cm, class_names):
    """打印詳細分析報告"""

    print("\n📊 分類報告：")
    print("-" * 70)
    print(classification_report(np.arange(len(class_names)),
                              np.arange(len(class_names)),
                              target_names=class_names,
                              output_dict=False,
                              zero_division=0))

    # 手動計算每類指標
    print("\n📈 每類詳細分析：")
    print(f"{'類別':<12} {'準確率':>10} {'召回率':>10} {'F1 分數':>10} {'樣本數':>10}")
    print("-" * 52)

    for i, class_name in enumerate(class_names):
        # 準確率（Precision）：預測為該類的中，有多少是真的
        # 預測為 A 的中，有多少真的是 A
        # TP / (TP + FN)
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0

        # 召回率（Recall）：真實為該類的中，有多少被預測對了
        # 真實為 A 的中，有多少被預測為 A
        # TP / (TP + FN)
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0

        # F1 分數
        # 準確率和召回率的調和平均
        # 2×P×R / (P+R)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 樣本數
        support = cm[i, :].sum()

        print(f"{class_name:<12} {precision*100:>9.2f}% {recall*100:>9.2f}% {f1*100:>9.2f}% {support:>10}")

    # 找出最容易混淆的類別對
    print("\n🔍 最容易混淆的類別對：")

    # 計算非對角線元素（錯誤預測）
    error_matrix = cm.copy()
    np.fill_diagonal(error_matrix, 0)

    # 找出最大的錯誤預測
    max_error = np.unravel_index(np.argmax(error_matrix), error_matrix.shape)
    true_class = class_names[max_error[0]]
    pred_class = class_names[max_error[1]]
    error_count = error_matrix[max_error]

    if error_count > 0:
        print(f"   {true_class} → {pred_class}: {error_count} 次")
        print(f"   💡 提示：這兩個類別視覺特徵可能相似")

def main():
    """主函數"""

    print("=" * 70)
    print(" " * 15 + "CIFAR-10 混淆矩陣分析")
    print("=" * 70)

    # ========== 1. 設置設備 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 設備：{device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ========== 2. 加載模型 ==========
    model_path = './models/best_cnn.pth'

    if not os.path.exists(model_path):
        print(f"\n❌ 模型文件不存在：{model_path}")
        print("💡 請先運行 train_cnn.py 訓練模型")
        return

    model = load_model(model_path, num_classes=10)
    model = model.to(device)

    # ========== 3. 加載測試數據 ==========
    print(f"\n📂 加載測試數據...")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0  # Windows 必須設為 0
    )

    print(f"✅ 測試集：{len(test_dataset)} 張圖像")

    # ========== 4. 獲取預測 ==========
    y_pred, y_true = get_predictions(model, test_loader, device)

    # ========== 5. 計算整體準確率 ==========
    accuracy = (y_pred == y_true).sum() / len(y_true) * 100
    print(f"\n✅ 預測完成！")
    print(f"   整體準確率：{accuracy:.2f}%")

    # ========== 6. 繪製混淆矩陣 ==========
    cm = plot_confusion_matrix(
        y_true, y_pred, CLASSES,
        save_path='./experiments/results/confusion_matrix.png',
        normalize=True  # 同時顯示計數和百分比
    )

    # ========== 7. 打印分析報告 ==========
    print_analysis(cm, CLASSES)

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ 分析完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()