"""
測試 Places365 數據集下載
"""
from datasets import load_dataset
import traceback

print("=" * 70)
print("測試 Places365 數據集連接")
print("=" * 70)

try:
    print("\n嘗試加載 Places365 數據集...")
    print("這可能需要幾分鐘...")

    # 嘗試加載小樣本
    dataset = load_dataset(
        'places365',
        split='train',
        streaming=True  # 使用流式加載測試
    )

    # 嘗試讀取第一個樣本
    first_example = next(iter(dataset))
    print("\n✅ 成功！")
    print(f"   數據集結構: {first_example.keys()}")
    print(f"   圖像模式: {first_example['image'].mode}")

except Exception as e:
    print(f"\n❌ 失敗: {e}")
    print("\n詳細錯誤信息：")
    traceback.print_exc()

    print("\n💡 可能的原因：")
    print("   1. Places365 數據集名稱不正確")
    print("   2. 需要特定權限")
    print("   3. 網絡問題")