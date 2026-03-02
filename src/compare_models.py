"""
對比 MLP 和 CNN 模型性能
"""

import torch
import matplotlib.pyplot as plt
import json
import os


def load_history(model_name):
    """加載訓練歷史"""
    path = f'./experiments/results/{model_name}_history.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def compare_models():
    """對比不同模型"""

    print("=" * 70)
    print(" " * 20 + "模型性能對比")
    print("=" * 70)

    # 加載歷史
    mlp_history = load_history('mlp')
    cnn_history = load_history('cnn')

    if mlp_history is None or cnn_history is None:
        print("⚠️  部分模型歷史不存在")
        print("💡 請先訓練 MLP 和 CNN 模型")
        print("   python train_cnn.py --model mlp --epochs 20")
        print("   python train_cnn.py --model cnn --epochs 20")
        return

    # 繪製對比圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 損失對比
    axes[0].plot(mlp_history['test_loss'], 'b--', label='MLP', linewidth=2)
    axes[0].plot(cnn_history['test_loss'], 'r-', label='CNN', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Test Loss', fontsize=12)
    axes[0].set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 準確率對比
    axes[1].plot(mlp_history['test_acc'], 'b--', label='MLP', linewidth=2)
    axes[1].plot(cnn_history['test_acc'], 'r-', label='CNN', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs('./experiments/results', exist_ok=True)
    plt.savefig('./experiments/results/model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ 對比圖表已保存：./experiments/results/model_comparison.png")
    plt.show()

    # 打印對比統計
    print("\n📊 性能對比：")
    print(f"   MLP 最佳準確率：{max(mlp_history['test_acc']):.2f}%")
    print(f"   CNN 最佳準確率：{max(cnn_history['test_acc']):.2f}%")
    print(f"   提升：{max(cnn_history['test_acc']) - max(mlp_history['test_acc']):.2f}%")


if __name__ == '__main__':
    compare_models()