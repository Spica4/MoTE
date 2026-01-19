"""
Evaluate Task 1 model on both Kaken and AMOS22 test sets.

This script loads the Task 1 checkpoint and performs task-specific evaluation:
- Task 0 classes (1-6) on Kaken test set
- Task 1 classes (7-12) on AMOS22 test set
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from utils.medical_data_manager import MedicalDataManager
from utils import factory

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename="evaluate_task1.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        args = json.load(f)
    return args

def main():
    # Setup logging
    setup_logging()

    # Load configuration
    config_path = "exps/medical/twostage_swin_unetr_task1.json"
    args = load_config(config_path)

    logging.info("=" * 80)
    logging.info("Task 1 Evaluation Script")
    logging.info("=" * 80)
    logging.info(f"Configuration: {config_path}")

    # Set device
    device = args["device"][0]
    args["device"] = [torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")]
    logging.info(f"Using device: {args['device'][0]}")

    # Create data manager
    logging.info("Creating data manager...")
    data_manager = MedicalDataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"][0],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    # Create model
    logging.info("Creating model...")
    model = factory.get_model(args["model_name"], args)

    # Load Task 1 checkpoint
    checkpoint_dir = "checkpoints/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], args["init_cls"], args["increment"]
    )
    checkpoint_path = os.path.join(checkpoint_dir, "task_1_checkpoint.pth")

    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        logging.error("Please train Task 1 first!")
        return

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args["device"][0])

    # Load model state
    model._network.load_state_dict(checkpoint["model_state_dict"])
    model._cur_task = checkpoint["task_id"]
    model._known_classes = checkpoint["known_classes"]
    model._total_classes = checkpoint["total_classes"]

    logging.info(f"Checkpoint loaded successfully")
    logging.info(f"Task ID: {model._cur_task}")
    logging.info(f"Known classes: {model._known_classes}")
    logging.info(f"Total classes: {model._total_classes}")

    # Move model to device
    model._network.to(args["device"][0])
    model._network.eval()

    # Prepare data manager for Task 1
    logging.info("Preparing data manager for Task 1...")
    data_manager.prepare_task(1)

    # Set data_manager in model (needed for evaluation)
    model.data_manager = data_manager

    # Create test loader for AMOS22 (default after prepare_task(1))
    test_classes = np.arange(0, model._total_classes)
    test_dataset = data_manager.get_dataset(
        test_classes, source="test", mode="test"
    )
    from torch.utils.data import DataLoader
    model.test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Run evaluation
    logging.info("\n" + "=" * 80)
    logging.info("Starting Task 1 Evaluation")
    logging.info("=" * 80)

    # The eval_task() method will automatically perform task-specific evaluation
    # for Task 1 (Kaken test set for classes 1-6, AMOS22 test set for classes 7-12)
    cnn_accy, _ = model.eval_task()

    # Print results
    logging.info("\n" + "=" * 80)
    logging.info("Final Results")
    logging.info("=" * 80)
    logging.info(f"Overall Dice: {cnn_accy['top1']:.2f}%")
    logging.info(f"Task-specific Dice scores:")
    for key, value in cnn_accy['grouped'].items():
        logging.info(f"  Classes {key}: {value:.2f}%")

    logging.info("\n" + "=" * 80)
    logging.info("Evaluation Complete!")
    logging.info("=" * 80)
    logging.info("Predictions saved to:")
    logging.info(f"  - predictions/task_1/task0_kaken/  (Task 0 predictions on Kaken)")
    logging.info(f"  - predictions/task_1/task1_amos22/ (Task 1 predictions on AMOS22)")
    logging.info("CSV results saved to:")
    logging.info(f"  - predictions/task_1/task0_kaken/evaluation_results_task0_kaken.csv")
    logging.info(f"  - predictions/task_1/task1_amos22/evaluation_results_task1_amos22.csv")

if __name__ == "__main__":
    main()
