"""
訓練結果可視化
生成損失曲線、準確率曲線等圖表
"""

import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

def main():
    print("=" * 70)
    print(" " * 20 + "訓練歷史可視化")
    print("=" * 70)

    # 加載歷史數據
    history_path = './experiments/results/cnn_history.json'

    if not os.path.exists(history_path):
        print(f"\n❌ 文件不存在：{history_path}")
        print("💡 請先運行 train_cnn.py 訓練模型")
        return

    print(f"\n📂 加載數據：{history_path}")
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    print(f"✅ 加載成功！")
    print(f"   Epochs: {len(history['train_loss'])}")

    # 創建圖表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ========== 左圖：損失曲線 ==========
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # ========== 右圖：準確率曲線 ==========
    axes[1].plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(history['test_acc'], 'r-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存圖表
    save_path = './experiments/results/training_history.png'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    print(f"\n✅ 圖表已保存：{save_path}")

    # 顯示圖表
    plt.show()

    # ========== 打印統計信息 ==========
    print("\n" + "=" * 70)
    print(" " * 20 + "訓練統計信息")
    print("=" * 70)

    print(f"\n📊 最終性能：")
    print(f"   最終訓練損失：{history['train_loss'][-1]:.4f}")
    print(f"   最終測試損失：{history['test_loss'][-1]:.4f}")
    print(f"   最終訓練準確率：{history['train_acc'][-1]:.2f}%")
    print(f"   最終測試準確率：{history['test_acc'][-1]:.2f}%")

    print(f"\n🏆 最佳性能：")
    best_epoch = history['test_acc'].index(max(history['test_acc'])) + 1
    print(f"   最佳測試準確率：{max(history['test_acc']):.2f}% (Epoch {best_epoch})")
    print(f"   對應測試損失：{history['test_loss'][best_epoch-1]:.4f}")

    print(f"\n📈 提升情況：")
    acc_improvement = history['test_acc'][-1] - history['test_acc'][0]
    print(f"   準確率提升：{acc_improvement:.2f}%")
    print(f"   損失降低：{history['test_loss'][0] - history['test_loss'][-1]:.4f}")

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ 可視化完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()