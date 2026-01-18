"""
Task 0学習後の推論画像保存スクリプト

Task 0のチェックポイントから推論を実行し、結果をnii.gz形式で保存します。
テストデータ、バリデーションデータ、または学習データで評価可能です。

使用例:
    # テストデータで評価・保存
    python evaluate_and_save_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth

    # バリデーションデータで評価・保存
    python evaluate_and_save_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth --data-split val

    # 保存先を指定
    python evaluate_and_save_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth --output-dir results/task0_predictions
"""

import argparse
import logging
import sys
import os
import torch
import numpy as np
import nibabel as nib
import csv
from torch.utils.data import DataLoader
from utils import factory
from utils.medical_data_manager import MedicalDataManager
from monai.inferers import sliding_window_inference


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_checkpoint(checkpoint_path, args):
    """チェックポイントからモデルをロード"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

    logging.info(f"チェックポイントをロード中: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # チェックポイントからargsを更新
    args.update(checkpoint.get("args", {}))

    # チェックポイントのクラス数でモデルを初期化
    # IMPORTANT: Use output layer size from state_dict to get correct number of classes
    # Sometimes checkpoint["total_classes"] doesn't match the actual output layer size
    checkpoint_state = checkpoint["model_state_dict"]
    if "backbone.base_model.out.conv.conv.weight" in checkpoint_state:
        output_layer_shape = checkpoint_state["backbone.base_model.out.conv.conv.weight"].shape
        checkpoint_total_classes = output_layer_shape[0]  # First dimension is number of classes
        logging.info(f"出力層のサイズから検出: {checkpoint_total_classes}クラス")
    else:
        checkpoint_total_classes = checkpoint.get("total_classes", 13)
        logging.info(f"チェックポイントのメタデータから読み取り: {checkpoint_total_classes}クラス")

    args["out_channels"] = checkpoint_total_classes
    logging.info(f"モデルを{checkpoint_total_classes}クラスで初期化")

    # モデル作成
    model = factory.get_model(args["model_name"], args)

    # state dictをロード（出力層のサイズは一致しているはず）
    model_state = model._network.state_dict()

    filtered_state = {}
    skipped_keys = []

    for key, value in checkpoint_state.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_state[key] = value
            else:
                skipped_keys.append(f"{key} (checkpoint: {value.shape}, model: {model_state[key].shape})")
        else:
            skipped_keys.append(f"{key} (現在のモデルに存在しません)")

    if skipped_keys:
        logging.warning(f"警告: {len(skipped_keys)}個のパラメータをスキップ:")
        for key in skipped_keys[:5]:
            logging.warning(f"  - {key}")
        if len(skipped_keys) > 5:
            logging.warning(f"  ... 他{len(skipped_keys) - 5}個")

    # フィルタリングしたstate dictをロード
    model._network.load_state_dict(filtered_state, strict=False)
    model._cur_task = checkpoint["task_id"]
    model._known_classes = checkpoint["known_classes"]
    model._total_classes = checkpoint["total_classes"]

    logging.info(f"Task {checkpoint['task_id']}のチェックポイントをロード完了")
    logging.info(f"既知クラス: {checkpoint['known_classes']}, 総クラス数: {checkpoint['total_classes']}")

    # デバッグ: 出力層の重みがロードされているか確認
    out_weight_key = "backbone.base_model.out.conv.conv.weight"
    out_bias_key = "backbone.base_model.out.conv.conv.bias"

    if out_weight_key in filtered_state:
        logging.info(f"✓ 出力層の重みをロード: {filtered_state[out_weight_key].shape}")
        # 重みの統計を表示
        weight_mean = filtered_state[out_weight_key].mean().item()
        weight_std = filtered_state[out_weight_key].std().item()
        logging.info(f"  重み統計 - 平均: {weight_mean:.6f}, 標準偏差: {weight_std:.6f}")
    else:
        logging.error(f"✗ 出力層の重みがロードされていません！")

    if out_bias_key in filtered_state:
        logging.info(f"✓ 出力層のバイアスをロード: {filtered_state[out_bias_key].shape}")
        # バイアスの値を表示
        bias_values = filtered_state[out_bias_key].cpu().numpy()
        logging.info(f"  バイアス値: {bias_values}")
    else:
        logging.error(f"✗ 出力層のバイアスがロードされていません！")

    return model, checkpoint


def compute_per_organ_dice(pred, target, num_classes=7):
    """
    臓器ごとのDice scoreを計算

    Args:
        pred: 予測マスク
        target: 正解マスク
        num_classes: クラス数（Task 0の場合は7: 背景0 + クラス1-6）

    Returns:
        クラスごとのDice scoreのリスト
    """
    dice_scores = []
    for class_id in range(num_classes):
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)

        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)

        if union == 0:
            dice = np.nan
        else:
            dice = (2.0 * intersection) / union

        dice_scores.append(dice)

    return dice_scores


def evaluate_and_save(model, data_manager, args, data_split="test", output_dir="predictions/task0"):
    """
    評価を実行し、推論結果を保存

    Args:
        model: 学習済みモデル
        data_manager: データマネージャー
        args: 引数辞書
        data_split: データ分割 ("train", "val", "test")
        output_dir: 出力ディレクトリ

    Returns:
        評価結果の辞書
    """
    device = args["device"][0]
    model._network.to(device)
    model._network.eval()

    # Kakenデータセットに強制的に切り替え
    if hasattr(data_manager._idata, 'current_stage'):
        data_manager._idata.current_stage = 0  # Kaken
        data_manager._train_data_dicts = data_manager._idata.kaken.train_data_dicts
        data_manager._val_data_dicts = data_manager._idata.kaken.val_data_dicts
        data_manager._test_data_dicts = data_manager._idata.kaken.test_data_dicts
        data_manager._train_trsf = data_manager._idata.kaken.train_trsf
        data_manager._test_trsf = data_manager._idata.kaken.test_trsf

    # データセット作成
    dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes), source=data_split, mode="test"
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logging.info(f"{data_split}データで評価中 ({len(dataset)}サンプル)...")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"推論結果の保存先: {output_dir}")

    # 評価設定
    roi_size = tuple(args.get("roi_size", [128, 128, 128]))
    sw_batch_size = args.get("sw_batch_size", 4)

    # 臓器名（Task 0: クラス1-6）
    organ_names = {
        0: "背景",
        1: "大動脈",
        2: "食道",
        3: "肝臓",
        4: "胆嚢",
        5: "胃",
        6: "脾臓",
    }

    all_preds = []
    all_targets = []
    csv_results = []

    with torch.no_grad():
        for batch_idx, (idx, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # スライディングウィンドウ推論
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=lambda x: model._network(x, test=True),
                overlap=0.5,
            )

            # 予測取得
            outputs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1, keepdim=True)

            # デバッグ: 最初の3サンプルの確率分布を表示
            if batch_idx < 3:
                # 中央付近のスライスの確率分布を確認
                center_z = outputs.shape[4] // 2
                center_y = outputs.shape[3] // 2
                center_x = outputs.shape[2] // 2
                center_probs = outputs[0, :, center_x, center_y, center_z].cpu().numpy()

                logging.info("=" * 60)
                logging.info(f"デバッグ: サンプル {batch_idx + 1} の中央ピクセルの確率分布")
                for class_id in range(len(center_probs)):
                    class_name = organ_names.get(class_id, f"クラス{class_id}")
                    logging.info(f"  {class_name:10s}: {center_probs[class_id]:.6f}")

                # クラスごとの予測ピクセル数を確認
                unique, counts = np.unique(preds.cpu().numpy(), return_counts=True)
                logging.info("-" * 60)
                logging.info("予測されたクラスの分布:")
                for cls, count in zip(unique, counts):
                    class_name = organ_names.get(int(cls), f"クラス{cls}")
                    percentage = count / preds.numel() * 100
                    logging.info(f"  {class_name:10s}: {count:8d} ピクセル ({percentage:5.2f}%)")
                logging.info("=" * 60)

            # numpy変換
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_preds.append(preds_np)
            all_targets.append(targets_np)

            # 元の画像パスを取得
            data_dict = loader.dataset.data_dicts[idx.item() if torch.is_tensor(idx) else idx]
            original_img_path = data_dict["image"]
            filename = os.path.basename(original_img_path)

            # 臓器ごとのDice score計算
            per_organ_dice = compute_per_organ_dice(preds_np[0], targets_np[0], num_classes=7)

            # CSV用の結果を保存
            result_row = {"filename": filename}
            for class_id in range(1, 7):  # クラス1-6のみ
                organ_name = organ_names[class_id]
                dice_value = per_organ_dice[class_id]
                result_row[organ_name] = dice_value if not np.isnan(dice_value) else 0.0

            # 平均Dice score計算（NaN除外）
            valid_dice_scores = [per_organ_dice[i] for i in range(1, 7) if not np.isnan(per_organ_dice[i])]
            avg_dice = np.mean(valid_dice_scores) if valid_dice_scores else 0.0
            result_row["平均"] = avg_dice

            csv_results.append(result_row)

            # nii.gz形式で保存
            original_img = nib.load(original_img_path)
            affine = original_img.affine
            header = original_img.header

            pred_data = preds_np[0, 0, :, :, :]
            pred_nifti = nib.Nifti1Image(pred_data.astype(np.int16), affine, header)

            pred_filename = filename.replace(".nii.gz", "_pred.nii.gz")
            pred_path = os.path.join(output_dir, pred_filename)
            nib.save(pred_nifti, pred_path)

            if (batch_idx + 1) % 5 == 0:
                logging.info(f"処理済み: {batch_idx + 1}/{len(loader)}サンプル")

    logging.info(f"{len(all_preds)}個の推論結果を{output_dir}に保存完了")

    # CSV保存
    csv_path = os.path.join(output_dir, f"evaluation_results_{data_split}.csv")
    fieldnames = ["filename"] + [organ_names[i] for i in range(1, 7)] + ["平均"]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_results)

    logging.info(f"評価結果を保存: {csv_path}")

    # 全体のDice score計算
    overall_dice = compute_overall_dice(all_preds, all_targets)

    return overall_dice, csv_results


def compute_overall_dice(y_pred_list, y_true_list):
    """
    全体のDice scoreを計算（Task 0のクラス1-6）

    Returns:
        辞書形式の結果
    """
    task0_classes = [1, 2, 3, 4, 5, 6]
    class_names = ["大動脈", "食道", "肝臓", "胆嚢", "胃", "脾臓"]

    # 全体のDice（クラス1-6統合）
    total_intersection = 0
    total_union = 0

    for y_pred, y_true in zip(y_pred_list, y_true_list):
        pred_mask = np.zeros_like(y_pred, dtype=bool)
        true_mask = np.zeros_like(y_true, dtype=bool)

        for cls in task0_classes:
            pred_mask |= (y_pred == cls)
            true_mask |= (y_true == cls)

        total_intersection += np.sum(pred_mask & true_mask)
        total_union += np.sum(pred_mask) + np.sum(true_mask)

    overall_dice = (2.0 * total_intersection) / total_union * 100 if total_union > 0 else 0.0

    # クラスごとのDice
    per_class = {}
    for cls, class_name in zip(task0_classes, class_names):
        cls_intersection = 0
        cls_union = 0

        for y_pred, y_true in zip(y_pred_list, y_true_list):
            pred_mask = (y_pred == cls)
            true_mask = (y_true == cls)

            cls_intersection += np.sum(pred_mask & true_mask)
            cls_union += np.sum(pred_mask) + np.sum(true_mask)

        cls_dice = (2.0 * cls_intersection) / cls_union * 100 if cls_union > 0 else 0.0
        per_class[class_name] = cls_dice

    return {"overall": overall_dice, "per_class": per_class}


def print_results(results, data_split):
    """評価結果を表示"""
    print("\n" + "=" * 60)
    print(f"Task 0 ({data_split}データ) 評価結果")
    print("=" * 60)
    print(f"全体Dice (クラス1-6): {results['overall']:.2f}%")
    print("-" * 60)
    print("クラス別Dice score:")
    for class_name, dice in results["per_class"].items():
        print(f"  {class_name:10s}: {dice:6.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Task 0学習後の推論画像保存スクリプト")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="チェックポイントファイルのパス (例: checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth)"
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="使用するデータ分割 (デフォルト: test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions/task0",
        help="出力ディレクトリ (デフォルト: predictions/task0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID (デフォルト: 0)"
    )

    args = parser.parse_args()

    setup_logging()

    # 基本設定
    eval_args = {
        "dataset": "twostage",
        "shuffle": False,
        "seed": 42,
        "init_cls": 6,
        "increment": 6,
        "model_name": "mote_seg",
        "backbone_type": "swin_unetr_mote",
        "device": [torch.device(f"cuda:{args.device}")],
        "img_size": [128, 128, 128],
        "in_channels": 1,
        "out_channels": 13,
        "feature_size": 48,
        "roi_size": [128, 128, 128],
        "sw_batch_size": 4,
        "ffn_num": 64,
        "ffn_adapter_init_option": "lora",
        "ffn_adapter_scalar": "0.1",
        "ffn_adapter_layernorm_option": "none",
    }

    # チェックポイントロード
    model, checkpoint = load_checkpoint(args.checkpoint, eval_args)

    # データマネージャー作成
    data_manager = MedicalDataManager(
        eval_args["dataset"],
        eval_args["shuffle"],
        eval_args["seed"],
        eval_args["init_cls"],
        eval_args["increment"],
        eval_args,
    )

    # 評価と保存
    results, csv_results = evaluate_and_save(
        model,
        data_manager,
        eval_args,
        data_split=args.data_split,
        output_dir=args.output_dir
    )

    # 結果表示
    print_results(results, args.data_split)

    # テキストファイルに保存
    result_file = os.path.join(args.output_dir, f"summary_{args.data_split}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Task 0 ({args.data_split}データ) 評価結果\n")
        f.write("=" * 60 + "\n")
        f.write(f"チェックポイント: {args.checkpoint}\n")
        f.write("-" * 60 + "\n")
        f.write(f"全体Dice (クラス1-6): {results['overall']:.2f}%\n")
        f.write("-" * 60 + "\n")
        f.write("クラス別Dice score:\n")
        for class_name, dice in results["per_class"].items():
            f.write(f"  {class_name:10s}: {dice:6.2f}%\n")
        f.write("=" * 60 + "\n")

    logging.info(f"評価サマリーを保存: {result_file}")
    logging.info("処理完了！")


if __name__ == "__main__":
    main()
