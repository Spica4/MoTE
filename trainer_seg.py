"""
Trainer for 3D medical image segmentation with incremental learning.

Modified from trainer.py to support segmentation tasks.
"""

import sys
import logging
import copy
import torch
import os
import numpy as np
from utils import factory
from utils.medical_data_manager import MedicalDataManager
from utils.toolkit import count_parameters


def train(args):
    """Main training entry point"""
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    """Training loop for incremental learning"""

    init_cls = args["init_cls"]
    logs_name = "logs_seg/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args['increment']
    )

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # Create checkpoint directory
    checkpoint_dir = "checkpoints/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"]
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logfilename = "logs_seg/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        "swin_unetr",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    # Create data manager for medical images
    data_manager = MedicalDataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    # Create model
    model = factory.get_model(args["model_name"], args)

    # Tracking metrics
    dice_curve = {"mean": [], "per_task": []}
    dice_matrix = []

    # Determine starting and ending task
    start_task = args.get("start_task", 0)
    end_task = args.get("end_task", data_manager.nb_tasks)  # Default: run all remaining tasks

    # Load checkpoint if resuming from a specific task
    if start_task > 0:
        checkpoint_path = os.path.join(
            checkpoint_dir, f"task_{start_task - 1}_checkpoint.pth"
        )
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model._network.load_state_dict(checkpoint["model_state_dict"])
            model._cur_task = checkpoint["task_id"]
            model._known_classes = checkpoint["known_classes"]
            model._total_classes = checkpoint["total_classes"]
            dice_curve = checkpoint.get("dice_curve", {"mean": [], "per_task": []})
            dice_matrix = checkpoint.get("dice_matrix", [])
            logging.info(f"Resumed from Task {start_task - 1}")
            logging.info(f"Known classes: {model._known_classes}, Total classes: {model._total_classes}")
        else:
            logging.warning(f"Checkpoint not found at {checkpoint_path}, starting from Task 0")
            start_task = 0

    logging.info(f"Will execute tasks from {start_task} to {end_task - 1} (inclusive)")

    # Train on each task
    for task in range(start_task, end_task):
        logging.info("=" * 50)
        logging.info(f"Starting Task {task}")
        logging.info("=" * 50)

        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        # Train on current task
        model.incremental_train(data_manager)

        # Evaluate
        dice_accy = model.eval_task()
        cnn_accy, _ = dice_accy if isinstance(dice_accy, tuple) else (dice_accy, None)

        # After task operations
        model.after_task()

        # Save checkpoint after each task
        checkpoint_path = os.path.join(checkpoint_dir, f"task_{task}_checkpoint.pth")
        checkpoint = {
            "task_id": model._cur_task,
            "known_classes": model._known_classes,
            "total_classes": model._total_classes,
            "model_state_dict": model._network.state_dict(),
            "dice_curve": dice_curve,
            "dice_matrix": dice_matrix,
            "args": args,
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        logging.info(f"Task {task} completed and saved")

        if cnn_accy is not None:
            logging.info("Dice scores: {}".format(cnn_accy["grouped"]))

            # Extract task-specific scores
            dice_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            dice_values = [cnn_accy["grouped"][key] for key in dice_keys]
            dice_matrix.append(dice_values)

            # Track mean Dice
            dice_curve["mean"].append(cnn_accy["top1"])
            dice_curve["per_task"].append(dice_values)

            logging.info("Mean Dice curve: {}".format(dice_curve["mean"]))
            logging.info("Average Dice: {}".format(
                sum(dice_curve["mean"]) / len(dice_curve["mean"])
            ))

    # Print final statistics
    if len(dice_matrix) > 0:
        np_dice_matrix = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(dice_matrix):
            idxy = len(line)
            np_dice_matrix[idxx, :idxy] = np.array(line)
        np_dice_matrix = np_dice_matrix.T

        logging.info('=' * 50)
        logging.info('Final Dice Matrix:')
        logging.info(np_dice_matrix)
        logging.info('=' * 50)

        # Compute forgetting
        forgetting = np.mean(
            (np.max(np_dice_matrix, axis=1) - np_dice_matrix[:, task])[:task]
        )
        logging.info('Forgetting: {:.4f}'.format(forgetting))
        logging.info('Final Average Dice: {:.4f}'.format(
            sum(dice_curve["mean"]) / len(dice_curve["mean"])
        ))


def _set_device(args):
    """Set device for training"""
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def print_args(args):
    """Print arguments"""
    logging.info("=" * 50)
    logging.info("Arguments:")
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
    logging.info("=" * 50)
