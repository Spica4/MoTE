# MoTE for 3D Medical Image Segmentation

このドキュメントでは、MoTE (Mixture of Task-specific Experts) フレームワークを3D医用画像セグメンテーションタスクに拡張した実装について説明します。

## 概要

このリポジトリは、2D画像分類向けのMoTEフレームワークを、3D医用画像のセグメンテーションタスク(nii.gz形式)に対応できるように拡張しています。MONAIフレームワークのSwin UNETRモデルを活用し、増分学習(Continual Learning)をサポートします。

## 主な機能

- **3D医用画像のサポート**: NIfTI (.nii.gz) 形式の医用画像に対応
- **Swin UNETR バックボーン**: MONAIのSwin UNETRを使用した最先端の3Dセグメンテーション
- **増分学習**: タスク固有のアダプターを使用した継続的学習
- **MONAI統合**: 医用画像処理に特化したMONAIライブラリの活用
- **柔軟なデータセット**: BraTS、腹部臓器、カスタムデータセットのサポート

## インストール

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

主な依存関係:
- PyTorch >= 2.0.0
- MONAI >= 1.3.0
- nibabel >= 5.0.0 (NIfTI画像読み込み用)
- timm >= 0.9.0

## データセット準備

### ディレクトリ構造

3D医用画像データセットは以下の構造で配置してください:

```
/datasets/your_dataset/
├── train/
│   ├── images/
│   │   ├── case_001.nii.gz
│   │   ├── case_002.nii.gz
│   │   └── ...
│   └── labels/
│       ├── case_001.nii.gz
│       ├── case_002.nii.gz
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

### サポートされているデータセット

1. **BraTS (Brain Tumor Segmentation)**
   - データセット名: `brats`
   - クラス数: 4 (背景、壊死、浮腫、増強領域)
   - 入力チャンネル: 4 (T1, T1ce, T2, FLAIR)

2. **Abdominal Organ Segmentation**
   - データセット名: `abdomen`
   - クラス数: カスタマイズ可能(デフォルト: 14)
   - 入力チャンネル: 1 (CT)

3. **Custom Medical 3D**
   - データセット名: `medical3d`
   - 任意の3D医用画像セグメンテーションタスクに対応

## 使用方法

### 1. 設定ファイルの作成

`exps/medical3d/` ディレクトリに設定ファイルのサンプルがあります:

- `swin_unetr_brats.json`: BraTSデータセット用
- `swin_unetr_abdomen.json`: 腹部臓器セグメンテーション用
- `swin_unetr_custom.json`: カスタムデータセット用テンプレート

### 2. 設定ファイルのカスタマイズ

```json
{
    "dataset": "medical3d",
    "data_dir": "/path/to/your/dataset",
    "init_cls": 2,
    "increment": 1,
    "model_name": "mote_seg",
    "backbone_type": "swin_unetr_mote_base",
    "num_classes": 5,
    "in_channels": 1,
    "roi_size": [96, 96, 96],
    "batch_size": 2,
    "init_epochs": 100,
    "init_lr": 0.0001,
    ...
}
```

重要なパラメータ:
- `dataset`: データセット名 (`brats`, `abdomen`, `medical3d`)
- `data_dir`: データセットのパス
- `init_cls`: 初期タスクのクラス数
- `increment`: 各増分タスクのクラス数
- `num_classes`: 総クラス数
- `roi_size`: パッチサイズ (3Dボリューム)
- `batch_size`: バッチサイズ(3D画像は大きいため小さめに設定)

### 3. トレーニングの実行

```bash
python main.py --config exps/medical3d/swin_unetr_custom.json
```

## アーキテクチャ

### モデル構造

```
SwinUNETRMoTE
├── Swin Transformer Encoder (バックボーン)
│   └── Task-specific Adapters (各レイヤー)
├── U-Net Decoder
└── Segmentation Head (クラス数に応じて拡張)
```

### アダプター方式

各タスクごとに:
1. 新しいアダプターセットを作成
2. バックボーンを凍結し、アダプターのみを訓練
3. タスク完了後、アダプターを保存
4. 次のタスク用に新しいアダプターを初期化

### 損失関数

- **DiceCELoss**: Dice Loss + Cross Entropy Loss の組み合わせ
- セグメンテーションタスクに最適化

## カスタムデータセットの追加

### 1. データセットクラスの作成

`utils/data_3d.py` に新しいクラスを追加:

```python
class YourCustomDataset(MedicalImage3D):
    def __init__(self, args: dict):
        super().__init__(args, roi_size=(96, 96, 96))
        self.num_classes = args.get("num_classes", 5)
        self.class_order = np.arange(self.num_classes).tolist()

    def download_data(self):
        data_dir = self.args.get("data_dir")
        train_images, train_labels = self.load_nifti_paths(data_dir, "train")
        test_images, test_labels = self.load_nifti_paths(data_dir, "test")

        self.train_data = self.create_data_dicts(train_images, train_labels)
        self.test_data = self.create_data_dicts(test_images, test_labels)

        self.train_targets = np.arange(len(self.train_data))
        self.test_targets = np.arange(len(self.test_data))
```

### 2. データマネージャーに登録

`utils/data_manager.py` の `_get_idata` 関数に追加:

```python
elif name == "your_dataset":
    return YourCustomDataset(args)
```

## 評価指標

### Dice Score (Dice係数)

セグメンテーション品質を測定する主要な指標:

```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```

- 範囲: 0 (完全不一致) ～ 1 (完全一致)
- クラスごとおよび平均Dice scoreを計算

## トラブルシューティング

### メモリ不足エラー

3D画像は大きいため、メモリエラーが発生する可能性があります:

**解決策**:
1. `batch_size` を小さくする (1 または 2)
2. `roi_size` を小さくする (例: `[64, 64, 64]`)
3. `sw_batch_size` を調整する
4. GPUメモリの大きいデバイスを使用

### データ読み込みエラー

**原因**: NIfTIファイルの形式やパスの問題

**解決策**:
1. ファイルパスが正しいか確認
2. nibabelで画像を読み込めるか確認:
   ```python
   import nibabel as nib
   img = nib.load("path/to/image.nii.gz")
   print(img.shape)
   ```

## 引用

このコードを使用する場合は、元のMoTE論文を引用してください:

```bibtex
@article{mote2024,
  title={MoTE: Mixture of Task-specific Experts for Class-Incremental Learning},
  author={...},
  journal={...},
  year={2024}
}
```

また、Swin UNETRとMONAIについても引用してください:

```bibtex
@article{hatamizadeh2022swin,
  title={Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger R and Xu, Daguang},
  journal={arXiv preprint arXiv:2201.01266},
  year={2022}
}

@article{cardoso2022monai,
  title={MONAI: An open-source framework for deep learning in healthcare},
  author={Cardoso, M Jorge and others},
  journal={arXiv preprint arXiv:2211.02701},
  year={2022}
}
```

## ライセンス

元のMoTEリポジトリのライセンスに従います。

## 貢献

バグ報告や機能リクエストは、GitHubのIssueで受け付けています。

## サポート

質問がある場合は、Issueを作成してください。

---

**注意**: この実装は研究目的で作成されています。臨床使用前には適切な検証が必要です。
