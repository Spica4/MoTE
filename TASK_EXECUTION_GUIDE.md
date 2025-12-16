# タスク別実行ガイド

このガイドでは、Task 0 と Task 1 を別々に実行する方法を説明します。

## 概要

インクリメンタル学習では、Task 0 の学習に長時間かかるため、Task 1 で問題が発生した場合に Task 0 から再実行する必要があるのは非効率です。

この機能により、各タスクを独立して実行でき、チェックポイントから再開できます。

---

## 使用方法

### **方法1: 全タスクを一度に実行（従来通り）**

```bash
python main_seg.py --config exps/medical/twostage_swin_unetr_debug.json
```

- Task 0 → Task 1 の順に自動実行
- 各タスク完了後にチェックポイント保存

---

### **方法2: Task 0 のみ実行**

```bash
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task0.json
```

**実行内容:**
- Task 0 (Kaken データセット) のみ学習
- 完了後、チェックポイントを保存
- Task 1 は実行しない

**保存場所:**
```
checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
```

---

### **方法3: Task 1 のみ実行（Task 0 のチェックポイントから再開）**

```bash
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task1.json
```

**実行内容:**
- Task 0 のチェックポイントを読み込み
- Task 1 (AMOS22 データセット) のみ学習
- Task 0 の学習結果を引き継ぐ

**必要なファイル:**
```
checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
```

---

## チェックポイントの内容

各チェックポイントには以下が保存されます:

```python
{
    "task_id": 0,                    # 完了したタスク ID
    "known_classes": 6,              # 学習済みクラス数
    "total_classes": 6,              # 現在の総クラス数
    "model_state_dict": {...},       # モデルの重み
    "dice_curve": {...},             # Dice score の履歴
    "dice_matrix": [...],            # タスクごとの Dice matrix
    "args": {...}                    # 実行時の設定
}
```

---

## 設定ファイルの違い

### `twostage_swin_unetr_debug.json`（全タスク実行）
```json
{
    "prefix": "medical_seg_debug",
    // start_task 未指定 (デフォルト: 0)
    ...
}
```

### `twostage_swin_unetr_debug_task0.json`（Task 0 のみ）
```json
{
    "prefix": "medical_seg_debug_task0",
    "start_task": 0,  // Task 0 から開始（実質 Task 0 のみ）
    ...
}
```

### `twostage_swin_unetr_debug_task1.json`（Task 1 のみ）
```json
{
    "prefix": "medical_seg_debug_task1",
    "start_task": 1,  // Task 1 から開始（Task 0 をスキップ）
    ...
}
```

---

## 実行例

### シナリオ1: Task 0 を実行して結果を確認

```bash
# Task 0 を実行
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task0.json

# 結果を確認
# Val Dice が良好なら Task 1 へ進む
```

### シナリオ2: Task 1 で失敗したので再実行

```bash
# Task 1 のみ再実行（Task 0 のチェックポイントから）
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task1.json

# Task 0 の学習は不要
```

### シナリオ3: ハイパーパラメータを変えて Task 1 を試す

```bash
# 1. Task 0 を実行（一度だけ）
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task0.json

# 2. Task 1 の設定を変更
# exps/medical/twostage_swin_unetr_debug_task1.json の
# ffn_num を 64 → 256 に変更

# 3. Task 1 を実行
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task1.json

# 4. 結果が良くなければ、別の設定で再度 Task 1 のみ実行
```

---

## 注意事項

1. **チェックポイントの互換性**
   - Task 0 のチェックポイントがないと Task 1 は実行できません
   - Task 1 を実行する前に、必ず Task 0 を完了してください

2. **設定の一貫性**
   - `init_cls`, `increment`, `seed` などの基本設定は統一してください
   - Task 1 で変更できるのは主に学習率、エポック数、アダプタサイズなどです

3. **チェックポイントの保存場所**
   - デフォルト: `checkpoints/mote_seg/twostage/6/6/`
   - `init_cls` や `increment` を変更すると保存場所も変わります

---

## トラブルシューティング

### Q: Task 1 で "Checkpoint not found" エラーが出る

**A:** Task 0 のチェックポイントが存在しません。
```bash
# Task 0 を先に実行
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task0.json
```

### Q: Task 1 の結果が悪い。Task 0 からやり直す必要がある？

**A:** 不要です。Task 1 の設定を変更して再実行してください。
```bash
# Task 1 の設定ファイルを編集
# 例: ffn_num を増やす、学習率を変える

# Task 1 のみ再実行
python main_seg.py --config exps/medical/twostage_swin_unetr_debug_task1.json
```

---

## まとめ

✅ **Task 0 と Task 1 を別々に実行可能**
✅ **チェックポイントから再開可能**
✅ **Task 1 の設定を変えて何度でも試せる**
✅ **時間を大幅に節約**
