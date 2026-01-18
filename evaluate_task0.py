"""
Evaluate model on Task 0 (Kaken) test dataset.

This script loads a checkpoint and evaluates it on Kaken test data
to measure Task 0 performance after Task 1 training (forgetting).

Usage:
    python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth
    python evaluate_task0.py --checkpoint checkpoints/mote_seg/twostage/6/6/task_0_checkpoint.pth
"""

import argparse
import logging
import sys
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from utils import factory
from utils.medical_data_manager import MedicalDataManager
from monai.inferers import sliding_window_inference


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_checkpoint(checkpoint_path, args):
    """Load model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Update args with checkpoint info
    args.update(checkpoint.get("args", {}))

    # IMPORTANT: Use checkpoint's total_classes to initialize model correctly
    # Task 0 checkpoint has 7 classes (0-6), Task 1 has 13 classes (0-12)
    checkpoint_total_classes = checkpoint.get("total_classes", 13)
    args["out_channels"] = checkpoint_total_classes
    logging.info(f"Initializing model with {checkpoint_total_classes} output classes")

    # Create model
    model = factory.get_model(args["model_name"], args)

    # Filter checkpoint state dict to handle size mismatches
    # (e.g., Task 0 checkpoint has 7 classes, but current model has 13)
    checkpoint_state = checkpoint["model_state_dict"]
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
            skipped_keys.append(f"{key} (not in current model)")

    if skipped_keys:
        logging.info(f"Skipped {len(skipped_keys)} parameters due to size mismatch or absence:")
        for key in skipped_keys[:5]:  # Show first 5
            logging.info(f"  - {key}")
        if len(skipped_keys) > 5:
            logging.info(f"  ... and {len(skipped_keys) - 5} more")

    # Load filtered state dict
    model._network.load_state_dict(filtered_state, strict=False)
    model._cur_task = checkpoint["task_id"]
    model._known_classes = checkpoint["known_classes"]
    model._total_classes = checkpoint["total_classes"]

    logging.info(f"Loaded checkpoint from Task {checkpoint['task_id']}")
    logging.info(f"Known classes: {checkpoint['known_classes']}, Total classes: {checkpoint['total_classes']}")

    return model, checkpoint


def evaluate_on_kaken(model, data_manager, args, save_predictions=False, pred_dir=None):
    """
    Evaluate model on Kaken (Task 0) test dataset.

    Args:
        model: Trained model
        data_manager: Data manager with Kaken dataset
        args: Arguments dict
        save_predictions: Whether to save predictions as nii.gz files
        pred_dir: Directory to save predictions (required if save_predictions=True)

    Returns:
        Dictionary with evaluation results
    """
    device = args["device"][0]
    model._network.to(device)
    model._network.eval()

    # Get Kaken test dataset (Task 0 classes: 1-6)
    # We need to temporarily switch to Kaken dataset
    # Force data_manager to use Kaken (stage 0)
    if hasattr(data_manager._idata, 'current_stage'):
        original_stage = data_manager._idata.current_stage
        data_manager._idata.current_stage = 0  # Force Kaken
        data_manager._train_data_dicts = data_manager._idata.kaken.train_data_dicts
        data_manager._val_data_dicts = data_manager._idata.kaken.val_data_dicts
        data_manager._test_data_dicts = data_manager._idata.kaken.test_data_dicts
        data_manager._train_trsf = data_manager._idata.kaken.train_trsf
        data_manager._test_trsf = data_manager._idata.kaken.test_trsf

    # Create test loader for Kaken dataset
    # Use all classes (0-12) for evaluation but only Kaken classes (1-6) exist
    test_dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes), source="test", mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logging.info(f"Evaluating on Kaken test set ({len(test_dataset)} samples)...")

    # Create prediction directory if saving
    if save_predictions:
        if pred_dir is None:
            raise ValueError("pred_dir must be specified when save_predictions=True")
        os.makedirs(pred_dir, exist_ok=True)
        logging.info(f"Predictions will be saved to: {pred_dir}")

    # Evaluation settings
    roi_size = tuple(args.get("roi_size", [128, 128, 128]))
    sw_batch_size = args.get("sw_batch_size", 4)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (idx, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Use sliding window inference
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=lambda x: model._network(x, test=True),
                overlap=0.5,
            )

            # Get predictions
            outputs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1, keepdim=True)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Save predictions as nii.gz if requested
            if save_predictions:
                # Get the original image path from the dataset
                data_dict = test_loader.dataset.data_dicts[idx.item() if torch.is_tensor(idx) else idx]
                original_img_path = data_dict["image"]

                # Load original image to get affine and header
                original_img = nib.load(original_img_path)
                affine = original_img.affine
                header = original_img.header

                # Create nifti image from prediction
                pred_data = preds.cpu().numpy()[0, 0, :, :, :]
                pred_nifti = nib.Nifti1Image(pred_data.astype(np.int16), affine, header)

                # Save prediction
                pred_filename = os.path.basename(original_img_path).replace(".nii.gz", "_pred.nii.gz")
                pred_path = os.path.join(pred_dir, pred_filename)
                nib.save(pred_nifti, pred_path)

            if (batch_idx + 1) % 5 == 0:
                logging.info(f"Processed {batch_idx + 1}/{len(test_loader)} samples")

    if save_predictions:
        logging.info(f"Saved {len(all_preds)} predictions to {pred_dir}")

    # Compute Dice scores for Task 0 classes (1-6)
    results = compute_task0_dice(all_preds, all_targets)

    # Restore original stage if needed
    if hasattr(data_manager._idata, 'current_stage'):
        data_manager._idata.current_stage = original_stage

    return results


def compute_task0_dice(y_pred_list, y_true_list):
    """
    Compute Dice scores for Task 0 classes (1-6).

    Args:
        y_pred_list: List of predicted masks
        y_true_list: List of ground truth masks

    Returns:
        Dictionary with Dice scores
    """
    results = {}

    # Task 0 classes: 1-6 (excluding background 0)
    task0_classes = [1, 2, 3, 4, 5, 6]
    class_names = ["大動脈", "食道", "肝臓", "胆嚢", "胃", "脾臓"]

    # Overall Dice for Task 0 (classes 1-6 combined)
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

    if total_union > 0:
        overall_dice = (2.0 * total_intersection) / total_union * 100
    else:
        overall_dice = 0.0

    results["overall"] = overall_dice
    results["per_class"] = {}

    # Per-class Dice
    for cls, class_name in zip(task0_classes, class_names):
        cls_intersection = 0
        cls_union = 0

        for y_pred, y_true in zip(y_pred_list, y_true_list):
            pred_mask = (y_pred == cls)
            true_mask = (y_true == cls)

            cls_intersection += np.sum(pred_mask & true_mask)
            cls_union += np.sum(pred_mask) + np.sum(true_mask)

        if cls_union > 0:
            cls_dice = (2.0 * cls_intersection) / cls_union * 100
        else:
            cls_dice = 0.0

        results["per_class"][class_name] = cls_dice

    return results


def print_results(results, checkpoint_path):
    """Print evaluation results"""
    print("\n" + "=" * 60)
    print("Task 0 (Kaken) Evaluation Results")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 60)
    print(f"Overall Dice (Classes 1-6): {results['overall']:.2f}%")
    print("-" * 60)
    print("Per-class Dice scores:")
    for class_name, dice in results["per_class"].items():
        print(f"  {class_name:10s}: {dice:6.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Task 0 (Kaken) test data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions as nii.gz files"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help="Directory to save predictions (default: predictions/task0_evaluation/)"
    )

    args = parser.parse_args()

    setup_logging()

    # Basic args
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

    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint, eval_args)

    # Create data manager
    data_manager = MedicalDataManager(
        eval_args["dataset"],
        eval_args["shuffle"],
        eval_args["seed"],
        eval_args["init_cls"],
        eval_args["increment"],
        eval_args,
    )

    # Setup prediction directory if saving
    pred_dir = None
    if args.save_predictions:
        if args.pred_dir is not None:
            pred_dir = args.pred_dir
        else:
            # Default: predictions/task0_evaluation/
            pred_dir = "predictions/task0_evaluation"
        logging.info(f"Predictions will be saved to: {pred_dir}")

    # Evaluate on Kaken test set
    results = evaluate_on_kaken(
        model,
        data_manager,
        eval_args,
        save_predictions=args.save_predictions,
        pred_dir=pred_dir
    )

    # Print results
    print_results(results, args.checkpoint)

    # Save results to file
    output_dir = os.path.dirname(args.checkpoint)
    output_file = os.path.join(output_dir, "task0_evaluation_results.txt")

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Task 0 (Kaken) Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Overall Dice (Classes 1-6): {results['overall']:.2f}%\n")
        f.write("-" * 60 + "\n")
        f.write("Per-class Dice scores:\n")
        for class_name, dice in results["per_class"].items():
            f.write(f"  {class_name:10s}: {dice:6.2f}%\n")
        f.write("=" * 60 + "\n")

    logging.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
