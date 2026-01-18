# Task 0評価ガイド

Task 0（Kakenデータセット）のテストデータで評価を行うためのガイドです。

## 目的

Task 1学習後に、Task 0のテストデータで評価することで、**真の忘却率**を測定できます。

現在の問題：
- Task 1学習後、AMOS22データセットのみで評価している
- AMOS22にはTask 0のクラス（1-6: 大動脈、食道、肝臓、胆嚢、胃、脾臓）が存在しない
- Task 0の性能変化（忘却）を正しく測定できていない

## 使用方法

### 基本的な使い方

```bash
python evaluate_task0.py --checkpoint <checkpoint_path>
```

### 例1: Task 0のチェックポイントを評価（ベースライン）

```bash
python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
```

**目的**: Task 0学習直後の性能を測定（ベースライン）

### 例2: Task 1のチェックポイントを評価（忘却測定）

```bash
python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth
```

**目的**: Task 1学習後のTask 0性能を測定（忘却を確認）

### GPU指定

```bash
python evaluate_task0.py --checkpoint <checkpoint_path> --device 1
```

### 推論結果を保存

```bash
python evaluate_task0.py --checkpoint <checkpoint_path> --save-predictions
```

推論結果は`predictions/task0_evaluation/`ディレクトリにnii.gz形式で保存されます。

### 保存先ディレクトリを指定

```bash
python evaluate_task0.py --checkpoint <checkpoint_path> --save-predictions --pred-dir custom_output_dir
```

## 出力例

```
============================================================
Task 0 (Kaken) Evaluation Results
============================================================
Checkpoint: checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth
------------------------------------------------------------
Overall Dice (Classes 1-6): 82.45%
------------------------------------------------------------
Per-class Dice scores:
  大動脈      :  85.32%
  食道        :  78.15%
  肝臓        :  88.91%
  胆嚢        :  75.43%
  胃          :  81.22%
  脾臓        :  85.67%
============================================================
```

## 忘却率の計算方法

### ステップ1: Task 0直後の性能を測定

```bash
python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
```

結果例: Overall Dice = **85.0%**

### ステップ2: Task 1学習後の性能を測定

```bash
python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth
```

結果例: Overall Dice = **82.5%**

### ステップ3: 忘却率を計算

```
忘却率 = Task 0直後のDice - Task 1学習後のDice
       = 85.0% - 82.5%
       = 2.5%
```

## 出力ファイル

### 評価結果（テキスト）

評価結果は自動的に保存されます：

```
checkpoints/mote_seg/twostage/6/6/task0_evaluation_results.txt
```

### 推論結果（nii.gz）

`--save-predictions`オプションを使用した場合、推論結果が保存されます：

```
predictions/task0_evaluation/
├── patient001_pred.nii.gz
├── patient002_pred.nii.gz
├── patient003_pred.nii.gz
...
```

各予測ファイルは元の画像と同じaffineとheaderを持ち、ITK-SNAPやMedSegなどのビューアで可視化できます。

## よくある質問

### Q: なぜこの評価が必要なのか？

**A:** 現在の評価では、Task 1学習後にAMOS22データのみで評価しています。AMOS22にはTask 0のクラスが存在しないため、Task 0の知識がどの程度保持されているかを測定できません。

### Q: Task 0の性能が下がっていたらどうすれば良いか？

**A:** 忘却率が高い（例: 10%以上）場合、以下の対策が考えられます：
- アダプタサイズを増やす（`ffn_num`: 64 → 128, 256）
- バックボーンの一部を解凍する
- タスク間のバランスを調整する

### Q: 理想的な忘却率は？

**A:**
- **0-3%**: 優秀 - ほとんど忘却なし
- **3-5%**: 良好 - わずかな忘却
- **5-10%**: 注意 - 明らかな忘却
- **10%以上**: 問題 - 大きな忘却

## 評価の流れ（推奨）

```bash
# 1. Task 0を学習
python main_seg.py --config exps/medical/twostage_swin_unetr_task0.json

# 2. Task 0の性能を測定（ベースライン）
python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
# → 結果をメモ（例: 85.0%）

# 3. Task 1を学習
python main_seg.py --config exps/medical/twostage_swin_unetr_task1.json

# 4. Task 1学習後のTask 0性能を測定
python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth
# → 結果をメモ（例: 82.5%）

# 5. 忘却率を計算
# 忘却率 = 85.0% - 82.5% = 2.5%
```

## まとめ

✅ Task 0のテストデータで正しく評価
✅ 真の忘却率を測定可能
✅ クラスごとの詳細な性能分析
✅ 結果を自動保存
