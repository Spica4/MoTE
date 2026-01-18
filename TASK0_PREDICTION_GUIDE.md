# Task 0推論画像保存ガイド

Task 0学習後に推論を実行し、結果をnii.gz形式で保存するためのガイドです。

## 概要

`evaluate_and_save_task0.py`は、Task 0のチェックポイントから推論を実行し、以下を保存します：

- **推論画像**: nii.gz形式（元の画像と同じaffine/header）
- **評価結果**: CSV形式（臓器ごとのDice score）
- **サマリー**: テキスト形式（全体および臓器別の統計）

## 基本的な使い方

### テストデータで評価・保存（デフォルト）

```bash
python evaluate_and_save_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
```

**出力先**: `predictions/task0/`

**出力ファイル**:
- `*_pred.nii.gz`: 推論画像（17個）
- `evaluation_results_test.csv`: 詳細な評価結果
- `summary_test.txt`: 評価サマリー

### バリデーションデータで評価・保存

```bash
python evaluate_and_save_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth \
  --data-split val
```

**出力先**: `predictions/task0/`

**データ数**: 18サンプル（Kakenバリデーションデータ）

### 学習データで評価・保存

```bash
python evaluate_and_save_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth \
  --data-split train
```

**出力先**: `predictions/task0/`

**データ数**: 54サンプル（Kaken学習データ）

### 出力ディレクトリを指定

```bash
python evaluate_and_save_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth \
  --output-dir results/task0_final
```

### GPU指定

```bash
python evaluate_and_save_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth \
  --device 1
```

## 出力ファイル詳細

### 1. 推論画像（nii.gz）

```
predictions/task0/
├── case001_pred.nii.gz
├── case002_pred.nii.gz
├── case003_pred.nii.gz
...
```

**特徴**:
- int16形式
- ラベル値: 0（背景）、1-6（臓器）
- 元の画像と同じ空間情報（affine/header）
- ITK-SNAP、3D Slicerなどで可視化可能

### 2. 評価結果CSV

`evaluation_results_test.csv`:

| filename | 大動脈 | 食道 | 肝臓 | 胆嚢 | 胃 | 脾臓 | 平均 |
|----------|--------|------|------|------|-----|------|------|
| case001.nii.gz | 0.85 | 0.78 | 0.89 | 0.75 | 0.82 | 0.86 | 0.825 |
| case002.nii.gz | 0.87 | 0.80 | 0.91 | 0.77 | 0.84 | 0.88 | 0.845 |
| ... | ... | ... | ... | ... | ... | ... | ... |

**内容**:
- 各行: 1つの画像ファイル
- 各列: 臓器ごとのDice score
- 最後の列: 6臓器の平均Dice score

### 3. 評価サマリー

`summary_test.txt`:

```
============================================================
Task 0 (testデータ) 評価結果
============================================================
チェックポイント: checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
------------------------------------------------------------
全体Dice (クラス1-6): 82.45%
------------------------------------------------------------
クラス別Dice score:
  大動脈      :  85.32%
  食道        :  78.15%
  肝臓        :  88.91%
  胆嚢        :  75.43%
  胃          :  81.22%
  脾臓        :  85.67%
============================================================
```

## 使用例：完全なワークフロー

### ステップ1: Task 0を学習

```bash
python main_seg.py --config exps/medical/twostage_swin_unetr_task0.json
```

### ステップ2: テストデータで評価・保存

```bash
python evaluate_and_save_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth \
  --data-split test \
  --output-dir predictions/task0_test
```

**結果**:
- 17個の推論画像
- Dice score: 全体およびクラス別
- CSV: 詳細な評価結果

### ステップ3: バリデーションデータでも評価

```bash
python evaluate_and_save_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth \
  --data-split val \
  --output-dir predictions/task0_val
```

**結果**:
- 18個の推論画像
- バリデーションデータでの性能評価

## データ分割の詳細

Kakenデータセット（全89サンプル）:

- **学習データ** (`--data-split train`): 54サンプル（インデックス0-53）
- **バリデーションデータ** (`--data-split val`): 18サンプル（インデックス54-71）
- **テストデータ** (`--data-split test`): 17サンプル（インデックス72-88）

## Task 0のクラス

| クラスID | 臓器名 |
|----------|--------|
| 0 | 背景 |
| 1 | 大動脈 |
| 2 | 食道 |
| 3 | 肝臓 |
| 4 | 胆嚢 |
| 5 | 胃 |
| 6 | 脾臓 |

## よくある質問

### Q: `evaluate_task0.py`との違いは？

**A:**

- **`evaluate_task0.py`**: Task 1学習後の忘却測定用（Kakenテストデータのみ）
- **`evaluate_and_save_task0.py`**: Task 0学習後の推論保存用（train/val/test選択可能）

### Q: 推論画像の確認方法は？

**A:** 以下のビューアで可視化できます：

```bash
# ITK-SNAPで開く
itksnap -g original_image.nii.gz -s prediction_pred.nii.gz

# 3D Slicerで開く
Slicer original_image.nii.gz prediction_pred.nii.gz
```

### Q: CSVファイルの使い方は？

**A:** Excelやpandasで開いて分析できます：

```python
import pandas as pd

# CSVを読み込み
df = pd.read_csv("predictions/task0/evaluation_results_test.csv")

# 平均Dice scoreで降順ソート
df_sorted = df.sort_values("平均", ascending=False)

# 統計情報
print(df["平均"].describe())

# 低性能の症例を抽出
low_performance = df[df["平均"] < 0.7]
```

### Q: 全データ分割で評価したい場合は？

**A:** シェルスクリプトで一括実行：

```bash
#!/bin/bash
CHECKPOINT="checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth"

# テストデータ
python evaluate_and_save_task0.py \
  --checkpoint $CHECKPOINT \
  --data-split test \
  --output-dir predictions/task0_test

# バリデーションデータ
python evaluate_and_save_task0.py \
  --checkpoint $CHECKPOINT \
  --data-split val \
  --output-dir predictions/task0_val

# 学習データ
python evaluate_and_save_task0.py \
  --checkpoint $CHECKPOINT \
  --data-split train \
  --output-dir predictions/task0_train

echo "全データ分割の評価完了！"
```

### Q: Task 1のチェックポイントでTask 0データを評価したい場合は？

**A:** `evaluate_task0.py`を使用してください：

```bash
# Task 1学習後のTask 0性能測定（忘却率計算用）
python evaluate_task0.py \
  --checkpoint checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth \
  --save-predictions
```

## まとめ

✅ Task 0学習後の推論結果を自動保存
✅ train/val/test全てのデータ分割に対応
✅ nii.gz + CSV + サマリーの3形式で出力
✅ ITK-SNAPなどで可視化可能
✅ CSVで詳細分析が可能

## 参考

- **Task 0とTask 1の両方を評価**: `trainer_seg.py`の自動評価機能
- **忘却率測定**: `evaluate_task0.py` + `EVALUATE_TASK0_GUIDE.md`
- **トレーニング**: `TASK_EXECUTION_GUIDE.md`
