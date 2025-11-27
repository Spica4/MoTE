"""
データセット検証スクリプト
3D医用画像セグメンテーションデータセットの構造と内容を検証
"""

import os
import nibabel as nib
import numpy as np
from pathlib import Path


def validate_dataset(data_dir, dataset_name="custom"):
    """
    データセットの構造と内容を検証

    Args:
        data_dir: データセットのルートディレクトリ
        dataset_name: データセット名
    """
    print("=" * 80)
    print(f"データセット検証: {dataset_name}")
    print(f"パス: {data_dir}")
    print("=" * 80)

    data_path = Path(data_dir)

    # ディレクトリ構造の確認
    print("\n[1] ディレクトリ構造の確認")
    required_dirs = [
        data_path / "train" / "images",
        data_path / "train" / "labels",
        data_path / "test" / "images",
        data_path / "test" / "labels",
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✓ {dir_path.relative_to(data_path)}")
        else:
            print(f"  ✗ {dir_path.relative_to(data_path)} が見つかりません！")
            return False

    # ファイル数の確認
    print("\n[2] ファイル数の確認")
    for split in ["train", "test"]:
        image_dir = data_path / split / "images"
        label_dir = data_path / split / "labels"

        image_files = sorted(list(image_dir.glob("*.nii.gz")))
        label_files = sorted(list(label_dir.glob("*.nii.gz")))

        print(f"  {split}:")
        print(f"    画像: {len(image_files)} ファイル")
        print(f"    ラベル: {len(label_files)} ファイル")

        if len(image_files) != len(label_files):
            print(f"    ⚠️  警告: 画像とラベルの数が一致しません！")

    # サンプルデータの詳細確認
    print("\n[3] サンプルデータの詳細確認")
    train_images = sorted(list((data_path / "train" / "images").glob("*.nii.gz")))
    train_labels = sorted(list((data_path / "train" / "labels").glob("*.nii.gz")))

    if len(train_images) > 0 and len(train_labels) > 0:
        # 最初のサンプルを確認
        sample_image = train_images[0]
        sample_label = train_labels[0]

        print(f"\n  サンプル画像: {sample_image.name}")

        try:
            # 画像の読み込み
            img = nib.load(str(sample_image))
            img_data = img.get_fdata()

            print(f"    形状: {img_data.shape}")
            print(f"    データ型: {img_data.dtype}")
            print(f"    値の範囲: {img_data.min():.2f} ~ {img_data.max():.2f}")
            print(f"    ボクセルサイズ: {img.header.get_zooms()[:3]}")

            # ラベルの読み込み
            lbl = nib.load(str(sample_label))
            lbl_data = lbl.get_fdata()

            print(f"\n  サンプルラベル: {sample_label.name}")
            print(f"    形状: {lbl_data.shape}")
            print(f"    データ型: {lbl_data.dtype}")

            unique_classes = np.unique(lbl_data)
            print(f"    クラス: {unique_classes}")
            print(f"    クラス数: {len(unique_classes)}")

            # クラスごとのボクセル数
            print(f"\n    クラスごとのボクセル数:")
            for cls in unique_classes:
                count = np.sum(lbl_data == cls)
                percentage = 100.0 * count / lbl_data.size
                print(f"      クラス {int(cls)}: {count:,} ({percentage:.2f}%)")

            # 形状の一致確認
            if img_data.shape != lbl_data.shape:
                print(f"\n    ⚠️  警告: 画像とラベルの形状が一致しません！")
                print(f"      画像: {img_data.shape}")
                print(f"      ラベル: {lbl_data.shape}")
            else:
                print(f"\n    ✓ 画像とラベルの形状が一致")

        except Exception as e:
            print(f"    ✗ エラー: {str(e)}")
            return False

    # メモリ要件の推定
    print("\n[4] メモリ要件の推定")
    if len(train_images) > 0:
        img = nib.load(str(train_images[0]))
        img_data = img.get_fdata()

        # 1サンプルのメモリ使用量
        sample_size_mb = img_data.nbytes / (1024 ** 2)
        print(f"  1サンプルのサイズ: {sample_size_mb:.2f} MB")

        # バッチサイズごとの推定
        for batch_size in [1, 2, 4]:
            batch_mb = sample_size_mb * batch_size * 2  # 画像+ラベル
            print(f"  バッチサイズ {batch_size}: 約 {batch_mb:.2f} MB")

        # ROIサイズでの推定
        print(f"\n  ROIパッチでのメモリ推定:")
        for roi_size in [(64, 64, 64), (96, 96, 96), (128, 128, 128)]:
            roi_voxels = np.prod(roi_size)
            roi_mb = roi_voxels * 4 / (1024 ** 2)  # float32想定
            for batch_size in [2, 4, 8]:
                total_mb = roi_mb * batch_size * 2
                print(f"    ROI {roi_size}, バッチ {batch_size}: 約 {total_mb:.2f} MB")

    print("\n" + "=" * 80)
    print("検証完了！")
    print("=" * 80)

    return True


def main():
    """メイン関数"""
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # デフォルトパス（カスタマイズしてください）
        data_dir = "/datasets/your_dataset"

    validate_dataset(data_dir)


if __name__ == "__main__":
    main()
