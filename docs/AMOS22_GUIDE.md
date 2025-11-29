# AMOS22データセット使用ガイド

AMOS22 (Abdominal Multi-Organ Segmentation 2022) データセットをMoTEフレームワークで使用するためのガイドです。

## データセット概要

AMOS22は、腹部臓器の3Dセグメンテーションデータセットで、CTとMRIの両方の画像を含んでいます。

### クラス情報（16クラス）

| ID | 臓器名（日本語） | 臓器名（英語） |
|----|--------------|--------------|
| 0  | 背景 | Background |
| 1  | 脾臓 | Spleen |
| 2  | 右腎臓 | Right Kidney |
| 3  | 左腎臓 | Left Kidney |
| 4  | 胆嚢 | Gallbladder |
| 5  | 食道 | Esophagus |
| 6  | 肝臓 | Liver |
| 7  | 胃 | Stomach |
| 8  | 大動脈 | Aorta |
| 9  | 下大静脈 | Inferior Vena Cava (IVC) |
| 10 | 膵臓 | Pancreas |
| 11 | 右副腎 | Right Adrenal Gland |
| 12 | 左副腎 | Left Adrenal Gland |
| 13 | 十二指腸 | Duodenum |
| 14 | 膀胱 | Bladder |
| 15 | 前立腺/子宮 | Prostate/Uterus |

## ディレクトリ構造

```
/datasets/amoss22/
├── imagesTr/       # 訓練画像 (200症例)
├── labelsTr/       # 訓練ラベル
├── imagesVa/       # 検証画像 (40症例)
├── labelsVa/       # 検証ラベル
├── imagesTs/       # テスト画像 (60症例)
└── labelsTs/       # テストラベル
```

**注意**: 実装はこのディレクトリ構造に対応しています。

## データセットの準備

### 1. データセットのダウンロード

AMOS22データセットは以下から入手できます:
- 公式サイト: https://amos22.grand-challenge.org/
- Zenodo: https://zenodo.org/record/7155725

### 2. ファイルの配置

ダウンロード後、以下のように配置してください:

```bash
# データセットを解凍
unzip amos22.zip -d /datasets/

# ディレクトリ構造の確認
ls -l /datasets/amoss22/
```

### 3. データの検証

データセットが正しく配置されているか確認:

```bash
python tools/validate_dataset.py /datasets/amoss22
```

**注意**: AMOS22は特殊な構造なので、このツールは一般的な構造を想定しています。エラーが出ても問題ありません。

### 4. クイック検証スクリプト

```python
from pathlib import Path
import nibabel as nib

data_dir = Path("/datasets/amoss22")

# ファイル数確認
train_imgs = list((data_dir / "imagesTr").glob("*.nii.gz"))
train_lbls = list((data_dir / "labelsTr").glob("*.nii.gz"))

print(f"訓練画像: {len(train_imgs)}")
print(f"訓練ラベル: {len(train_lbls)}")

# サンプルデータの確認
if train_imgs:
    img = nib.load(str(train_imgs[0]))
    lbl = nib.load(str(train_lbls[0]))
    print(f"\n画像形状: {img.shape}")
    print(f"ラベル形状: {lbl.shape}")
```

## 使用方法

### 1. 設定ファイルの使用

提供されている設定ファイルを使用:

```bash
python main.py --config exps/medical3d/swin_unetr_amos22.json
```

### 2. 設定のカスタマイズ

`exps/medical3d/swin_unetr_amos22.json` を編集:

```json
{
    "dataset": "amos22",
    "data_dir": "/datasets/amoss22",
    "init_cls": 4,      // 初期タスクのクラス数
    "increment": 3,     // 増分タスクのクラス数
    "num_classes": 16,  // 総クラス数
    "roi_size": [96, 96, 96],  // パッチサイズ
    "batch_size": 2,    // バッチサイズ
    "use_validation": true  // 検証セットをテストに使用
}
```

### 3. タスク分割例

設定例（`init_cls=4`, `increment=3`）の場合:

- **Task 0**: クラス 0, 1, 2, 3 (背景, 脾臓, 右腎臓, 左腎臓)
- **Task 1**: クラス 4, 5, 6 を追加 (胆嚢, 食道, 肝臓)
- **Task 2**: クラス 7, 8, 9 を追加 (胃, 大動脈, 下大静脈)
- **Task 3**: クラス 10, 11, 12 を追加 (膵臓, 右副腎, 左副腎)
- **Task 4**: クラス 13, 14, 15 を追加 (十二指腸, 膀胱, 前立腺/子宮)

## パラメータ推奨設定

### GPU メモリ別の推奨設定

| GPU メモリ | ROI サイズ | バッチサイズ | SW バッチサイズ |
|-----------|-----------|------------|---------------|
| 12GB      | [64, 64, 64] | 1 | 2 |
| 16GB      | [96, 96, 96] | 2 | 4 |
| 24GB      | [128, 128, 128] | 2-4 | 4-8 |
| 32GB+     | [128, 128, 128] | 4-8 | 8-16 |

### 訓練時間の目安

- **1エポック**: 約30-60分（データ数とGPUに依存）
- **初期タスク（150エポック）**: 約75-150時間
- **増分タスク（100エポック）**: 約50-100時間

## トラブルシューティング

### メモリ不足エラー

```json
{
    "roi_size": [64, 64, 64],  // より小さく
    "batch_size": 1,           // 1に減らす
    "sw_batch_size": 2         // 小さく
}
```

### データ読み込みエラー

ファイル名の一致を確認:
```bash
ls /datasets/amoss22/imagesTr/ | head -5
ls /datasets/amoss22/labelsTr/ | head -5
```

画像とラベルのファイル名（拡張子を除く）が一致している必要があります。

### CUDA Out of Memory

以下を試してください:
1. バッチサイズを1に減らす
2. ROIサイズを小さくする
3. `sw_batch_size`を減らす
4. グラディエントチェックポイントを有効化（実装要）

## データ特性

### CTとMRIの混在

AMOS22にはCTとMRIの両方が含まれています。モダリティを分けて学習することも可能です:

```python
# ファイル名からモダリティを判別
# AMOS22では通常、ファイル名にモダリティ情報が含まれています
```

### クラス不均衡

一部の臓器（例：副腎、胆嚢）は小さく、不均衡があります。
必要に応じて重み付けを調整してください。

## 評価メトリクス

- **Dice係数**: 各臓器および平均
- **ハウスドルフ距離**: より詳細な境界評価に使用可能（実装要）

## 引用

AMOS22を使用する場合は、以下を引用してください:

```bibtex
@article{ji2022amos,
  title={AMOS: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation},
  author={Ji, Yuanfeng and Bai, Haotian and Yang, Jie and Ge, Chongjian and Zhu, Ye and Zhang, Ruimao and Li, Zhen and Zhang, Lingyan and Ma, Wanling and Wan, Xiang and others},
  journal={arXiv preprint arXiv:2206.08023},
  year={2022}
}
```

## 参考リンク

- AMOS22 Challenge: https://amos22.grand-challenge.org/
- データセット論文: https://arxiv.org/abs/2206.08023
- Leaderboard: https://amos22.grand-challenge.org/evaluation/challenge/leaderboard/

---

**注意**: このガイドは研究目的での使用を想定しています。臨床応用には適切な検証が必要です。
