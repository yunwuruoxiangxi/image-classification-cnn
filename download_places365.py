"""
從 MIT 官方服務器下載 Places365
官方網站：http://places2.csail.mit.edu/download.html
"""
import os
import requests
from pathlib import Path
import tarfile
from tqdm import tqdm


def download_file(url, filepath):
    """下載文件並顯示進度"""
    print(f"\n📥 下載: {os.path.basename(filepath)}")
    print(f"   URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f, tqdm(
                desc="   進度",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✅ 下載完成: {filepath}")
        return True

    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return False


def main():
    print("=" * 70)
    print(" " * 15 + "Places365 官方下載工具")
    print("=" * 70)
    print("\n官方網站: http://places2.csail.mit.edu/download.html")
    print("\n注意：")
    print("  - 需要註冊賬號（免費）")
    print("  - 文件較大（約 12GB），請確保有足夠空間")
    print("  - 首次下載可能需要較長時間")

    # 確認
    confirm = input("\n是否繼續下載？(y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消")
        return

    # 創建目錄
    data_dir = Path('./data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)

    # 下載 URL
    base_url = "http://places2.csail.mit.edu"

    # 1. 下載數據（選擇小版本）
    print("\n" + "=" * 70)
    print("選擇下載版本：")
    print("  1. Places365 Standard (256x256, ~12GB) - 推薦")
    print("  2. Places365 Challenge (227x227, ~42GB)")

    choice = input("請選擇 (1/2): ").strip()

    if choice == '1':
        data_url = f"{base_url}/places365_standard.tar"
        data_file = data_dir / "places365_standard.tar"
    else:
        data_url = f"{base_url}/places365_train.tar"
        data_file = data_dir / "places365_train.tar"

    # 檢查是否已存在
    if data_file.exists():
        print(f"\n✅ 文件已存在: {data_file}")
        print(f"   大小: {data_file.stat().st_size / (1024 ** 3):.2f} GB")
    else:
        # 下載
        success = download_file(data_url, data_file)
        if not success:
            print("\n❌ 下載失敗，請檢查網絡連接")
            print("💡 提示：可能需要科學上網")
            return

    # 2. 下載類別標籤
    print("\n📥 下載類別標籤...")
    labels_url = f"{base_url}/categories_places365.txt"
    labels_file = data_dir / "categories_places365.txt"

    if not labels_file.exists():
        download_file(labels_url, labels_file)
    else:
        print(f"✅ 標籤文件已存在")

    print("\n" + "=" * 70)
    print("✅ 數據集下載完成！")
    print("=" * 70)
    print(f"\n文件位置：")
    print(f"  數據: {data_file}")
    print(f"  標籤: {labels_file}")
    print("\n下一步：")
    print("  解壓數據並組織到 train/val/test 文件夾")


if __name__ == '__main__':
    main()