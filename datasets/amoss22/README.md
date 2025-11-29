# AMOS22 Dataset Setup

このディレクトリにAMOS22データセットを配置してください。

## ディレクトリ構造

以下の構造でデータを配置してください:

```
datasets/amoss22/
├── imagesTr/       # 訓練画像 (.nii.gz)
├── labelsTr/       # 訓練ラベル (.nii.gz)
├── imagesVa/       # 検証画像 (.nii.gz)
├── labelsVa/       # 検証ラベル (.nii.gz)
├── imagesTs/       # テスト画像 (.nii.gz)
└── labelsTs/       # テストラベル (.nii.gz)
```

## データセットの取得

1. **AMOS22 公式サイト**: https://amos22.grand-challenge.org/
2. **Zenodo**: https://zenodo.org/record/7155725

## セットアップ手順

### 1. データセットのダウンロード

```bash
# Zenodoまたは公式サイトからダウンロード
# 例: wget https://zenodo.org/record/7155725/files/amos22.zip
```

### 2. 解凍と配置

```bash
# このディレクトリに解凍
cd /path/to/MoTE/datasets/amoss22
unzip /path/to/amos22.zip

# または直接解凍
unzip /path/to/amos22.zip -d /path/to/MoTE/datasets/amoss22/
```

### 3. 構造の確認

```bash
# ファイル数の確認
ls imagesTr/*.nii.gz | wc -l  # 訓練画像数
ls labelsTr/*.nii.gz | wc -l  # 訓練ラベル数
ls imagesVa/*.nii.gz | wc -l  # 検証画像数
ls labelsVa/*.nii.gz | wc -l  # 検証ラベル数
```

期待される数:
- 訓練: 200症例
- 検証: 40症例
- テスト: 60症例

### 4. データの検証

```bash
# プロジェクトルートから実行
cd /path/to/MoTE
python tools/validate_dataset.py ./datasets/amoss22
```

## 注意事項

- **データサイズ**: 約100GB以上
- **フォーマット**: NIfTI (.nii.gz)
- **画像タイプ**: CTスキャン
- **クラス数**: 16 (背景 + 15臓器)

## トレーニング実行

データセットの準備が完了したら:

```bash
# テスト実行
python main.py --config exps/medical3d/swin_unetr_amos22_test.json

# 本番実行
python main.py --config exps/medical3d/swin_unetr_amos22.json
```

## トラブルシューティング

### ディレクトリが見つからない

```bash
# 現在のディレクトリ構造を確認
ls -la datasets/amoss22/
```

### ファイルが見つからない

```bash
# 各ディレクトリの内容を確認
ls datasets/amoss22/imagesTr/ | head
ls datasets/amoss22/labelsTr/ | head
```

## 参考

詳細は `docs/AMOS22_GUIDE.md` を参照してください。
