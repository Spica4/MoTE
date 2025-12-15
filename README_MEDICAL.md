# MoTE for 3D Medical Image Segmentation

このドキュメントでは、MoTE (Mixture of Task-specific Experts) を3D医療画像セグメンテーションに適用する方法を説明します。

## 概要

このリポジトリには、Swin UNETRモデルにMoTEフレームワークを統合した3D医療画像セグメンテーション用のコードが含まれています。2段階の漸進学習をサポートしています：

- **Stage 1**: Kakenデータセット（6クラス: 1-6）
- **Stage 2**: AMOS22データセット（6クラス: 7-12）

## 主要な機能

- **Swin UNETR**: 3D医療画像セグメンテーション用の最先端アーキテクチャ
- **MoTE Adapters**: Transformerブロックに統合されたタスク固有のアダプター
- **2段階学習**: 異なるデータセットでの逐次学習
- **Expert Selection**: テスト時の信頼度ベースの専門家選択
- **MONAI統合**: 医療画像処理用の包括的な前処理パイプライン

## 必要な環境

```bash
pip install torch torchvision
pip install monai[all]
pip install timm
pip install easydict
pip install tqdm
```

## データセット構造

### Kakenデータセット
```
/deeparea/sokabe/Dataset/Kaken_set_A/torso/1.5mm/
├── image/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── label/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

- **総数**: 89サンプル
- **分割**:
  - Train: 0-53 (54サンプル)
  - Val: 54-71 (18サンプル)
  - Test: 72-88 (17サンプル)
- **ラベルマッピング**:
  - Original: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
  - Fixed: [0,0,0,1,2,3,4,5,6,0,0,0,0,0,0,0,0]
  - 使用: 1-6 (6クラス)

### AMOS22データセット
```
/deeparea/sokabe/Dataset/amos22/amos22/
├── imagesTr/
│   ├── amos_0001.nii.gz
│   ├── amos_0002.nii.gz
│   └── ...
└── labelsTr/
    ├── amos_0001.nii.gz
    ├── amos_0002.nii.gz
    └── ...
```

- **総数**: 200サンプル
- **分割**:
  - Train: 0-119 (120サンプル)
  - Val: 120-159 (40サンプル)
  - Test: 160-199 (40サンプル)
- **ラベルマッピング**:
  - Original: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  - Fixed: [0,7,8,0,0,0,0,0,9,10,0,0,0,11,12]
  - 使用: 7-12 (6クラス)

## 前処理パイプライン

### 訓練時の前処理
1. **LoadImaged**: NIfTI画像の読み込み
2. **EnsureChannelFirstd**: チャンネル次元を最初に配置
3. **MapLabelValued**: ラベルの再マッピング
4. **Orientationd**: RAS方向への統一
5. **NormalizeIntensityd**: 強度の正規化
6. **CropForegroundd**: 前景のクロップ
7. **Spacingd**: 1.5mm等方性リサンプリング
8. **SpatialPadd**: 128³にパディング
9. **RandCropByPosNegLabeld**: ランダムクロップ（4サンプル/画像）
10. **RandFlipd**: ランダム反転（各軸10%）
11. **RandShiftIntensityd**: 強度シフト（50%確率）
12. **Lambdad**: クラスフィルタリング
13. **ToTensord**: テンソルへの変換

### バリデーション/テスト時の前処理
訓練時の前処理からデータ拡張（ランダムクロップ、反転、強度シフト）を除いたもの

## モデルアーキテクチャ

### Swin UNETR with MoTE

```
SwinUNETR(
  - swinViT (Swin Transformer encoder)
    - layers1-4 (各層に2つのTransformerブロック)
      - SwinTransformerBlock (with MoTE Adapter)
        - WindowAttention
        - MLPBlock
        - Adapter (タスク固有)
  - encoder1-4 (UnetrBasicBlock)
  - decoder1-5 (UnetrUpBlock)
  - out (UnetOutBlock)
)
```

### MoTE Adapter

各TransformerブロックにAdapterを挿入：
- **Down projection**: dim → bottleneck (64)
- **Non-linearity**: ReLU
- **Up projection**: bottleneck → dim
- **Residual connection**: 元の特徴に加算

## 使用方法

### 訓練の実行

```bash
python main_seg.py --config ./exps/medical/twostage_swin_unetr.json
```

### 設定ファイルのパラメータ

`exps/medical/twostage_swin_unetr.json`:

```json
{
    "dataset": "twostage",           # 2段階データセット
    "init_cls": 6,                   # Stage 1のクラス数
    "increment": 6,                  # Stage 2のクラス数
    "model_name": "mote_seg",        # モデル名

    "init_epochs": 100,              # Stage 1のエポック数
    "later_epochs": 100,             # Stage 2のエポック数
    "init_lr": 0.0001,               # 学習率
    "batch_size": 2,                 # バッチサイズ
    "optimizer": "adamw",            # オプティマイザ
    "scheduler": "cosine",           # スケジューラ

    "img_size": [128, 128, 128],     # 入力サイズ
    "in_channels": 1,                # 入力チャンネル数
    "out_channels": 13,              # 出力クラス数（背景含む）
    "feature_size": 48,              # 基本特徴次元
    "roi_size": [128, 128, 128],     # ROIサイズ
    "sw_batch_size": 4,              # スライディングウィンドウバッチサイズ

    "ffn_num": 64,                   # Adapterのボトルネック次元
    "ffn_adapter_scalar": "0.1"      # Adapterのスケーリング係数
}
```

## 2段階学習の流れ

### Stage 1: Kakenデータセット
1. Swin UNETRの初期化
2. Stage 1用のAdapterの作成
3. Kakenデータセット（クラス1-6）で訓練
4. Adapterの凍結とリストへの保存

### Stage 2: AMOS22データセット
1. 出力層を13クラス（0-12）に更新
2. Stage 2用の新しいAdapterの作成
3. AMOS22データセット（クラス7-12）で訓練
4. テスト時にExpert Selection（専門家選択）を使用

## Expert Selection（専門家選択）

テスト時に、各専門家（Adapter）の予測を組み合わせる：

1. **各専門家で予測**: 各Adapterで独立に予測を生成
2. **信頼度計算**: 各専家の予測のsoftmax最大値から信頼度を計算
3. **重み付け**: 最高信頼度の専門家に大きな重みを割り当て、他は0.1倍
4. **統合**: 重み付けされた予測を組み合わせる

```python
# 最高信頼度の専門家: weight = confidence
# その他の専門家: weight = confidence * 0.1
# Softmax正規化後、加重平均
```

## 評価指標

### Dice係数
- **Per-class Dice**: 各クラスのDice係数
- **Mean Dice**: 全クラスの平均Dice係数
- **Task-specific Dice**: 各タスク（Stage 1/2）のDice係数

### Forgetting
```
Forgetting = mean(max(Dice_task_over_time) - Dice_task_final)
```

## ログと出力

訓練ログは `logs_seg/mote_seg/twostage/6/6/` に保存されます：
- 訓練ログ
- Dice曲線
- Dice行列
- Forgetting指標

## ファイル構造

```
MoTE/
├── backbone/
│   └── swin_unetr_mote.py          # Swin UNETR + MoTE
├── models/
│   ├── base_seg.py                 # セグメンテーション用ベースクラス
│   └── mote_seg.py                 # セグメンテーション用MoTEモデル
├── utils/
│   ├── medical_data.py             # 医療画像データセット
│   ├── medical_data_manager.py     # データマネージャー
│   ├── inc_net_seg.py              # セグメンテーション用ネットワーク
│   └── factory.py                  # モデルファクトリー
├── exps/
│   └── medical/
│       └── twostage_swin_unetr.json  # 設定ファイル
├── main_seg.py                     # メインエントリーポイント
├── trainer_seg.py                  # トレーナー
└── README_MEDICAL.md               # このファイル
```

## トラブルシューティング

### メモリ不足エラー
- `batch_size` を小さくする（1または2推奨）
- `sw_batch_size` を小さくする
- `num_workers` を減らす

### データセットが見つからない
- データセットパスを確認:
  - Kaken: `/deeparea/sokabe/Dataset/Kaken_set_A/torso/1.5mm`
  - AMOS22: `/deeparea/sokabe/Dataset/amos22/amos22`
- ディレクトリ構造を確認（`image/`と`label/`フォルダが存在するか）

### CUDA out of memory
- より小さい画像サイズを使用（例: [96, 96, 96]）
- Mixed precision訓練を有効化（実装が必要）
- より小さいモデル（`feature_size=24`）を使用

## 参考文献

- **Swin UNETR**: Tang et al., "Self-supervised pre-training of swin transformers for 3d medical image analysis," CVPR 2022
- **MoTE**: "Mixture of Task-specific Experts for Pre-Trained Model Based Class-incremental Learning"
- **MONAI**: https://monai.io/

## ライセンス

このプロジェクトは元のMoTEリポジトリと同じライセンスに従います。
