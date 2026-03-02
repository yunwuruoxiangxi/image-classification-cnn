"""
數據加載與預處理模塊
負責下載和處理 Places365 數據集
"""

import os
from pathlib import Path
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import yaml

# 推薦的 20 個場景類別
SCENE_CATEGORIES = [
    'bedroom', 'kitchen', 'office', 'classroom', 'restaurant',
    'living_room', 'bathroom', 'airport_terminal', 'gym', 'library',
    'street', 'forest', 'beach', 'mountain', 'lake',
    'park', 'river', 'skyscraper', 'bridge', 'subway_station'
]


def load_config(config_path='config.yaml'):
    """加載 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def download_places365_huggingface(data_dir='./data/raw',
                                    categories=None,
                                    split='train'):
    """
    從 HuggingFace 下載 Places365 數據集

    Args:
        data_dir: 數據保存目錄
        categories: 場景類別列表
        split: 'train' 或 'validation'

    Returns:
        dataset: HuggingFace Dataset 或 None
    """
    if categories is None:
        categories = SCENE_CATEGORIES

    print("=" * 70)
    print(" " * 20 + "Places365 數據集下載")
    print("=" * 70)
    print(f"📥 準備下載...")
    print(f"   場景類別: {len(categories)} 類")
    print(f"   數據集劃分: {split}")
    print(f"   保存路徑: {data_dir}")

    try:
        # 導入 datasets 庫
        from datasets import load_dataset

        print("\n🔗 正在連接 HuggingFace...")
        print("   首次使用可能需要登錄：huggingface-cli login")

        # 加載 Places365 數據集
        dataset = load_dataset(
            'places365',
            split=split,
            cache_dir=os.path.join(data_dir, '.cache')
        )

        print(f"\n✅ 原始數據集下載成功！")
        print(f"   總圖像數量: {len(dataset)}")

        # 獲取類別標籤
        if hasattr(dataset, 'features') and 'label' in dataset.features:
            all_labels = dataset.features['label'].names
            print(f"   總類別數: {len(all_labels)}")
        else:
            print("   無法獲取類別信息")
            return None

        # 過濾出需要的類別
        print(f"\n🔍 正在過濾目標類別...")
        target_indices = []
        for idx, label_name in enumerate(all_labels):
            if label_name in categories:
                target_indices.append(idx)

        print(f"   找到 {len(target_indices)} 個目標類別")

        # 過濾數據集
        target_indices_set = set(target_indices)
        filtered_dataset = dataset.filter(
            lambda x: x['label'] in target_indices_set
        )

        print(f"✅ 過濾完成！")
        print(f"   保留圖像: {len(filtered_dataset)} 張")

        return filtered_dataset

    except ImportError:
        print("\n❌ 錯誤：datasets 庫未安裝！")
        print("💡 請運行：pip install datasets huggingface_hub")
        return None

    except Exception as e:
        print(f"\n❌ 下載失敗: {e}")
        print("\n💡 可能的解決方案：")
        print("   1. 首次使用需要登錄：huggingface-cli login")
        print("   2. 檢查網絡連接")
        print("   3. 獲取 token：https://huggingface.co/settings/tokens")
        return None


def save_dataset_to_folders(dataset, output_dir='./data/processed', image_size=128):
    """
    將 HuggingFace 數據集保存到文件夾結構

    結構：
    data/processed/
        train/
            bedroom/
                000001.jpg
            kitchen/
                ...
        val/
        test/
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "保存數據集到文件夾")
    print("=" * 70)

    # 獲取類別名稱映射
    if not hasattr(dataset, 'features') or 'label' not in dataset.features:
        print("❌ 無法獲取類別信息")
        return False

    label_names = dataset.features['label'].names

    # 創建目錄結構
    print(f"📁 創建目錄結構...")
    for split in ['train', 'val', 'test']:
        for category in SCENE_CATEGORIES:
            dir_path = Path(output_dir) / split / category
            dir_path.mkdir(parents=True, exist_ok=True)

    # 圖像轉換
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
    ])

    # 隨機劃分比例
    random.seed(42)

    print(f"\n💾 正在保存圖像...")
    print(f"   目標尺寸: {image_size}x{image_size}")
    print(f"   劃分比例: 80% train, 10% val, 10% test")

    # 統計計數
    stats = {'train': 0, 'val': 0, 'test': 0}

    for idx, example in enumerate(dataset):
        # 獲取圖像和標籤
        image = example['image']
        label_idx = example['label']
        category_name = label_names[label_idx]

        # 只處理目標類別
        if category_name not in SCENE_CATEGORIES:
            continue

        # 轉換為 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 調整尺寸
        image = transform(image)

        # 隨機決定劃分
        rand_val = random.random()
        if rand_val < 0.8:
            split = 'train'
        elif rand_val < 0.9:
            split = 'val'
        else:
            split = 'test'

        # 保存圖像
        save_path = Path(output_dir) / split / category_name / f'{idx:06d}.jpg'
        image.save(save_path, quality=95)
        stats[split] += 1

        # 進度顯示
        if (idx + 1) % 1000 == 0:
            print(f"   已處理 {idx + 1} 張圖像...")

    print(f"\n✅ 保存完成！")
    print(f"   Train: {stats['train']} 張")
    print(f"   Val:   {stats['val']} 張")
    print(f"   Test:  {stats['test']} 張")
    print(f"   總計:  {sum(stats.values())} 張")
    print(f"   保存路徑: {output_dir}")

    return True


def create_data_loaders(data_dir='./data/processed', batch_size=32, image_size=128):
    """
    創建 PyTorch DataLoader

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "創建數據加載器")
    print("=" * 70)

    # 數據增強（訓練集）
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 驗證和測試集（不做增強）
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 加載數據集
    print(f"📂 從 {data_dir} 加載數據集...")

    try:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )

        test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'test'),
            transform=val_transform
        )

        # 獲取類別名稱
        class_names = train_dataset.classes

        print(f"✅ 數據集加載成功！")
        print(f"   類別數: {len(class_names)}")
        print(f"   類別: {', '.join(class_names[:5])}...")

    except FileNotFoundError:
        print(f"\n❌ 錯誤：找不到數據集目錄 {data_dir}")
        print("💡 請先運行數據下載腳本！")
        return None, None, None, None

    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n📊 數據加載器統計：")
    print(f"   Train: {len(train_dataset)} 張圖像, {len(train_loader)} 個批次")
    print(f"   Val:   {len(val_dataset)} 張圖像, {len(val_loader)} 個批次")
    print(f"   Test:  {len(test_dataset)} 張圖像, {len(test_loader)} 個批次")

    return train_loader, val_loader, test_loader, class_names


def main():
    """主函數：執行完整數據準備流程"""
    print("\n" + "=" * 70)
    print(" " * 15 + "Places365 場景分類 - 數據準備")
    print("=" * 70)

    # 加載配置
    try:
        config = load_config('config.yaml')
        print("✅ 配置文件加載成功")
    except FileNotFoundError:
        print("⚠️  未找到 config.yaml，使用默認配置")
        config = {}

    # 設置路徑
    raw_dir = config.get('data', {}).get('raw_dir', './data/raw')
    processed_dir = config.get('data', {}).get('processed_dir', './data/processed')
    image_size = config.get('data', {}).get('image_size', 128)
    batch_size = config.get('training', {}).get('batch_size', 32)

    # Step 1: 下載數據集
    dataset = download_places365_huggingface(
        data_dir=raw_dir,
        categories=SCENE_CATEGORIES,
        split='train'
    )

    if dataset is None:
        print("\n❌ 數據集下載失敗，終止流程")
        return

    # Step 2: 保存到文件夾
    success = save_dataset_to_folders(
        dataset=dataset,
        output_dir=processed_dir,
        image_size=image_size
    )

    if not success:
        print("\n❌ 數據保存失敗，終止流程")
        return

    # Step 3: 創建數據加載器
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=processed_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    if train_loader is None:
        print("\n❌ 數據加載器創建失敗")
        return

    # Step 4: 測試數據加載
    print("\n" + "=" * 70)
    print(" " * 20 + "測試數據加載")
    print("=" * 70)

    try:
        images, labels = next(iter(train_loader))
        print(f"✅ 測試成功！")
        print(f"   批次形狀: images={images.shape}, labels={labels.shape}")
        print(f"   類別名稱示例: {class_names[labels[0].item()]}")
    except Exception as e:
        print(f"⚠️  測試失敗: {e}")

    print("\n" + "=" * 70)
    print(" " * 20 + "🎉 數據準備完成！")
    print("=" * 70)
    print("\n下一步：")
    print("  1. 運行 Jupyter Notebook 進行數據探索")
    print("  2. 開始編寫和訓練模型")
    print()


if __name__ == '__main__':
    main()