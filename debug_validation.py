"""
Debug script to investigate why Val Dice is 0.0000 during Task 1 training.

This script checks:
1. Validation dataset labels (are there any classes 7-12?)
2. Model predictions (is it predicting only background?)
3. Dice metric computation
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from utils.medical_data_manager import MedicalDataManager
from utils import factory
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s => %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

def analyze_labels(loader, num_samples=5):
    """Analyze label distribution in validation data"""
    logging.info("\n" + "=" * 80)
    logging.info("Analyzing Validation Data Labels")
    logging.info("=" * 80)

    class_counts = np.zeros(13, dtype=np.int64)

    for idx, (_, inputs, targets) in enumerate(loader):
        if idx >= num_samples:
            break

        targets_np = targets.cpu().numpy()
        unique, counts = np.unique(targets_np, return_counts=True)

        logging.info(f"\nSample {idx + 1}:")
        logging.info(f"  Shape: {targets_np.shape}")
        logging.info(f"  Unique classes: {unique}")
        logging.info(f"  Distribution:")

        for cls, count in zip(unique, counts):
            if cls < 13:
                class_counts[int(cls)] += count
            percentage = count / targets_np.size * 100
            class_name = get_class_name(int(cls))
            logging.info(f"    Class {int(cls)} ({class_name}): {count} pixels ({percentage:.2f}%)")

    logging.info("\n" + "=" * 80)
    logging.info(f"Overall class distribution (first {num_samples} samples):")
    total_pixels = class_counts.sum()
    for cls in range(13):
        if class_counts[cls] > 0:
            percentage = class_counts[cls] / total_pixels * 100
            class_name = get_class_name(cls)
            logging.info(f"  Class {cls} ({class_name}): {class_counts[cls]} pixels ({percentage:.2f}%)")

    # Check if any Task 1 classes exist
    task1_pixels = class_counts[7:13].sum()
    if task1_pixels == 0:
        logging.warning("\n⚠️  WARNING: No Task 1 classes (7-12) found in validation data!")
        logging.warning("This would cause Val Dice to be 0.0000")

    return class_counts

def analyze_predictions(model, loader, device, num_samples=3):
    """Analyze model predictions"""
    logging.info("\n" + "=" * 80)
    logging.info("Analyzing Model Predictions")
    logging.info("=" * 80)

    model.eval()

    with torch.no_grad():
        for idx, (_, inputs, targets) in enumerate(loader):
            if idx >= num_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=(128, 128, 128),
                sw_batch_size=4,
                predictor=lambda x: model(x, test=False),
                overlap=0.5,
            )

            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Analyze
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            unique_pred, counts_pred = np.unique(preds_np, return_counts=True)
            unique_target, counts_target = np.unique(targets_np, return_counts=True)

            logging.info(f"\nSample {idx + 1}:")
            logging.info(f"  Prediction shape: {preds_np.shape}")
            logging.info(f"  Predicted classes: {unique_pred}")
            logging.info(f"  Prediction distribution:")
            for cls, count in zip(unique_pred, counts_pred):
                percentage = count / preds_np.size * 100
                class_name = get_class_name(int(cls))
                logging.info(f"    Class {int(cls)} ({class_name}): {percentage:.2f}%")

            logging.info(f"  Ground truth classes: {unique_target}")

            # Check probability distribution
            max_probs = torch.max(probs, dim=1)[0]
            mean_confidence = max_probs.mean().item()
            logging.info(f"  Mean prediction confidence: {mean_confidence:.4f}")

            # Check if predicting only background
            bg_percentage = (preds == 0).sum().item() / preds.numel() * 100
            if bg_percentage > 99.0:
                logging.warning(f"  ⚠️  Model predicting {bg_percentage:.2f}% background!")

def get_class_name(cls):
    """Get organ name for class"""
    names = [
        "背景", "大動脈", "食道", "肝臓", "胆嚢", "胃", "脾臓",
        "右腎臓", "左腎臓", "下大動脈", "膵臓", "膀胱", "子宮"
    ]
    return names[cls] if cls < len(names) else f"Class{cls}"

def main():
    setup_logging()

    logging.info("=" * 80)
    logging.info("Validation Debug Script for Task 1")
    logging.info("=" * 80)

    # Load config
    config_path = "exps/medical/twostage_swin_unetr_task1.json"
    with open(config_path, 'r') as f:
        args = json.load(f)

    # Set device
    device = torch.device(f"cuda:{args['device'][0]}" if torch.cuda.is_available() else "cpu")
    args["device"] = [device]

    logging.info(f"Device: {device}")

    # Create data manager
    data_manager = MedicalDataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"][0],
        args["init_cls"],
        args["increment"],
        args,
    )

    # Prepare for Task 1
    data_manager.prepare_task(1)

    # Get validation dataset
    val_classes = np.arange(0, 12)
    val_dataset = data_manager.get_dataset(
        val_classes, source="val", mode="test"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    logging.info(f"Validation dataset size: {len(val_dataset)} samples")

    # Analyze validation labels
    class_counts = analyze_labels(val_loader, num_samples=5)

    # Load model
    logging.info("\n" + "=" * 80)
    logging.info("Loading Task 1 checkpoint...")
    logging.info("=" * 80)

    checkpoint_dir = "checkpoints/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], args["init_cls"], args["increment"]
    )
    checkpoint_path = os.path.join(checkpoint_dir, "task_1_checkpoint.pth")

    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint not found: {checkpoint_path}")
        logging.info("Skipping model prediction analysis")
        return

    # Create model
    model = factory.get_model(args["model_name"], args)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model._network.load_state_dict(checkpoint["model_state_dict"])
    model._network.to(device)
    model._network.eval()

    logging.info("Checkpoint loaded successfully")

    # Analyze predictions
    analyze_predictions(model._network, val_loader, device, num_samples=3)

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("Summary")
    logging.info("=" * 80)

    task1_pixels = class_counts[7:13].sum()
    total_pixels = class_counts.sum()

    if task1_pixels == 0:
        logging.error("❌ ISSUE FOUND: Validation data contains NO Task 1 classes (7-12)")
        logging.error("   This is why Val Dice = 0.0000")
        logging.error("   The DiceMetric with include_background=False ignores background,")
        logging.error("   and there are no foreground classes to compute Dice on.")
    elif task1_pixels / total_pixels < 0.01:
        logging.warning("⚠️  Task 1 classes make up less than 1% of validation data")
        logging.warning("   This could cause very low or unstable Dice scores")
    else:
        logging.info(f"✓ Task 1 classes found: {task1_pixels / total_pixels * 100:.2f}% of pixels")

if __name__ == "__main__":
    main()
