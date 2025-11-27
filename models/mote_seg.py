"""
MoTE Learner for 3D Medical Image Segmentation

This module implements incremental learning for 3D medical image segmentation
using the Mixture of Task-specific Experts (MoTE) approach.
"""
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from utils.inc_net_3d import SegMoteNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 8


class SegmentationLearner(BaseLearner):
    """
    Incremental learner for 3D medical image segmentation

    This learner handles continual learning of segmentation tasks,
    where each task may introduce new anatomical structures or organs.
    """

    def __init__(self, args: Dict):
        """
        Args:
            args: Configuration dictionary
        """
        super().__init__(args)

        self._network = SegMoteNet(args, pretrained=True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 1e-5)
        self.min_lr = args.get("min_lr", 1e-6)
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        # Segmentation-specific parameters
        self.roi_size = args.get("roi_size", (96, 96, 96))
        self.sw_batch_size = args.get("sw_batch_size", 4)  # For sliding window inference
        self.overlap = args.get("overlap", 0.5)

        # Loss function
        self.use_dice_ce = args.get("use_dice_ce", True)
        if self.use_dice_ce:
            self.criterion = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=True,
                smooth_nr=0.0,
                smooth_dr=1e-6,
            )
        else:
            self.criterion = DiceLoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=True,
            )

        # Metrics
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False,
        )

    def after_task(self):
        """Update known classes after task completion"""
        self._known_classes = self._total_classes

    def get_cls_range(self, task_id: int):
        """
        Get class range for a specific task

        Args:
            task_id: Task identifier

        Returns:
            Tuple of (start_class, end_class)
        """
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def incremental_train(self, data_manager):
        """
        Train on a new incremental task

        Args:
            data_manager: Data manager providing training/test data
        """
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_seg_head(self._total_classes)

        logging.info(f"Learning segmentation on classes {self._known_classes}-{self._total_classes}")

        self.data_manager = data_manager

        # Get training data for new classes only
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Get test data for all classes seen so far
        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test"
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # For sliding window inference
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Handle multi-GPU training
        if len(self._multiple_gpus) > 1:
            print("Using multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Freeze and save adapter
        self._network.freeze_backbone()
        self._network.add_adapter()

    def _train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Main training loop

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
        """
        self._network.to(self._device)

        # Determine learning rate and epochs based on task
        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
            epochs = self.args["init_epochs"]
        else:
            lr = self.args.get("later_lr", self.args["init_lr"])
            epochs = self.args.get("later_epochs", self.args["init_epochs"])
            optimizer = self.get_optimizer(lr=lr)
            scheduler = self.get_scheduler(optimizer, epochs)

        self._init_train(train_loader, test_loader, optimizer, scheduler, epochs)

    def get_optimizer(self, lr: float):
        """
        Create optimizer

        Args:
            lr: Learning rate

        Returns:
            Optimizer
        """
        optimizer_name = self.args.get("optimizer", "adamw").lower()

        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def get_scheduler(self, optimizer, epochs: int):
        """
        Create learning rate scheduler

        Args:
            optimizer: Optimizer
            epochs: Number of epochs

        Returns:
            Scheduler or None
        """
        scheduler_name = self.args.get("scheduler", "cosine")

        if scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=epochs,
                eta_min=self.min_lr
            )
        elif scheduler_name == "steplr":
            milestones = self.args.get("init_milestones", [30, 60, 90])
            gamma = self.args.get("init_lr_decay", 0.1)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=milestones,
                gamma=gamma
            )
        elif scheduler_name == "constant":
            scheduler = None
        else:
            scheduler = None

        return scheduler

    def _init_train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer,
        scheduler,
        epochs: int
    ):
        """
        Training loop implementation

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            optimizer: Optimizer
            scheduler: LR scheduler
            epochs: Number of epochs
        """
        best_dice = 0.0
        prog_bar = tqdm(range(epochs))

        for epoch in prog_bar:
            self._network.train()

            epoch_loss = 0.0
            step = 0

            for batch_data in train_loader:
                step += 1

                # Handle MONAI dictionary format
                if isinstance(batch_data, dict):
                    inputs = batch_data["image"].to(self._device)
                    labels = batch_data["label"].to(self._device)
                else:
                    # Handle tuple format (idx, inputs, labels)
                    _, inputs, labels = batch_data
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

                # Map labels to relative indices for current task
                # Only train on new classes (old classes are background)
                labels_relative = labels.clone()
                labels_relative = torch.where(
                    labels_relative >= self._known_classes,
                    labels_relative - self._known_classes,
                    0,  # Map old classes to background
                )

                # Forward pass
                output = self._network(inputs, test=False)
                logits = output["logits"]

                # Compute loss
                loss = self.criterion(logits, labels_relative)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            epoch_loss /= step
            current_lr = optimizer.param_groups[0]["lr"]

            # Validation
            if (epoch + 1) % self.args.get("val_interval", 5) == 0:
                dice_score = self._validate(test_loader)

                if dice_score > best_dice:
                    best_dice = dice_score

                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {epoch_loss:.4f}, Dice {dice_score:.4f}, "
                    f"Best Dice {best_dice:.4f}, LR {current_lr:.6f}"
                )
            else:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {epoch_loss:.4f}, LR {current_lr:.6f}"
                )

            prog_bar.set_description(info)
            logging.info(info)

    def _validate(self, test_loader: DataLoader) -> float:
        """
        Validation step

        Args:
            test_loader: Test data loader

        Returns:
            Mean Dice score
        """
        self._network.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for batch_data in test_loader:
                # Handle MONAI dictionary format
                if isinstance(batch_data, dict):
                    inputs = batch_data["image"].to(self._device)
                    labels = batch_data["label"].to(self._device)
                else:
                    _, inputs, labels = batch_data
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

                # Sliding window inference for large volumes
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=lambda x: self._network(x, test=True)["logits"],
                    overlap=self.overlap,
                )

                # Get predictions
                outputs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1, keepdim=True)

                # Compute Dice score
                self.dice_metric(y_pred=predictions, y=labels)

        # Get mean Dice score
        dice_score = self.dice_metric.aggregate().item()

        return dice_score

    def _eval_task(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on test set

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of metrics
        """
        dice_score = self._validate(test_loader)

        return {
            "dice": dice_score,
        }

    def eval_task(self):
        """Evaluate current task"""
        metrics = self._eval_task(self.test_loader)
        logging.info(f"Task {self._cur_task} evaluation: Dice = {metrics['dice']:.4f}")
        return metrics
