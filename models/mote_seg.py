"""
MoTE for 3D Medical Image Segmentation

Implements incremental learning for medical image segmentation using
Swin UNETR with task-specific adapters.
"""

import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base_seg import BaseSegLearner
from utils.toolkit import tensor2numpy
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import copy

num_workers = 4  # Reduced for 3D medical images (memory intensive)


class SegLearner(BaseSegLearner):
    """
    Learner for 3D medical image segmentation with MoTE.

    Key differences from classification MoTE:
    - Uses segmentation losses (Dice + CE)
    - Evaluates with Dice coefficient per class
    - Uses sliding window inference for large volumes
    - Handles dense predictions instead of single labels
    """

    def __init__(self, args):
        super().__init__(args)

        # Import here to avoid circular dependency
        from utils.inc_net_seg import MoteSegNet

        self._network = MoteSegNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]
        self.adapter_num = args.get("adapter_num", -1)

        # Segmentation-specific settings
        self.roi_size = args.get("roi_size", (128, 128, 128))
        self.sw_batch_size = args.get("sw_batch_size", 4)  # Sliding window batch size

        # Loss function for segmentation
        self.criterion = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            reduction="mean",
        )

        # Metrics for evaluation
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean_batch",
            get_not_nans=False,
        )

    def after_task(self):
        self._known_classes = self._total_classes

    def get_cls_range(self, task_id):
        """Get class range for a given task"""
        if task_id == 0:
            start_cls = 1  # Skip background (0)
            end_cls = self.init_cls + 1
        else:
            start_cls = self.init_cls + 1 + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def incremental_train(self, data_manager):
        """Train on a new task"""
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # Prepare data manager for current task (switches dataset for stage 2)
        data_manager.prepare_task(self._cur_task)

        # Update network output channels
        self._network.update_fc(self._total_classes + 1)  # +1 for background

        logging.info("Learning on classes {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager

        # Get dataset for current task
        # For task 0: classes 1-6, for task 1: classes 7-12
        task_classes = np.arange(self._known_classes, self._total_classes)

        self.train_dataset = data_manager.get_dataset(
            task_classes, source="train", mode="train"
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # Batch size 1 for sliding window inference
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Train
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Freeze and add adapter to list
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def _train(self, train_loader, test_loader):
        """Main training loop"""
        self._network.to(self._device)

        # Get epochs and learning rate
        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, epochs)
        else:
            epochs = self.args.get('later_epochs', self.args['init_epochs'])
            lr = self.args.get("later_lr", self.args["init_lr"])
            optimizer = self.get_optimizer(lr=lr)
            scheduler = self.get_scheduler(optimizer, epochs)

        self._init_train(train_loader, test_loader, optimizer, scheduler, epochs)

    def get_optimizer(self, lr):
        """Get optimizer for training"""
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args['optimizer']}")

        return optimizer

    def get_scheduler(self, optimizer, epochs):
        """Get learning rate scheduler"""
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=epochs, eta_min=self.min_lr
            )
        elif self.args["scheduler"] == 'steplr':
            milestones = self.args.get("init_milestones", [60, 120, 170])
            gamma = self.args.get("init_lr_decay", 0.1)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=gamma
            )
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.args['scheduler']}")

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, epochs):
        """Training loop for current task"""
        prog_bar = tqdm(range(epochs))

        for epoch in prog_bar:
            self._network.train()

            losses = 0.0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # inputs: [B, C, H, W, D]
                # targets: [B, 1, H, W, D] with class indices
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Forward pass
                logits = self._network(inputs, test=False)

                # Compute loss
                loss = self.criterion(logits, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

            if scheduler is not None:
                scheduler.step()

            # Validation
            if (epoch + 1) % self.args.get("val_interval", 5) == 0:
                mean_dice = self._validate(test_loader)
                logging.info(
                    "Task {}, Epoch {}/{} => Loss {:.3f}, Val Dice {:.4f}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        mean_dice,
                    )
                )

            info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _validate(self, test_loader):
        """Validate model on test set"""
        self._network.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for _, inputs, targets in test_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # Use sliding window inference for large volumes
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=lambda x: self._network(x, test=False),
                    overlap=0.5,
                )

                # Compute Dice metric
                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                self.dice_metric(y_pred=outputs, y=targets)

        # Get mean Dice score
        mean_dice = self.dice_metric.aggregate().item()
        return mean_dice

    def _eval_cnn(self, loader):
        """
        Evaluate model using segmentation metrics.

        Returns predictions and ground truth for Dice computation.
        """
        self._network.eval()
        self.dice_metric.reset()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # Use sliding window inference
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=lambda x: self._network(x, test=True),
                    overlap=0.5,
                )

                # Get predictions
                outputs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1, keepdim=True)

                # Compute Dice
                self.dice_metric(y_pred=preds, y=targets)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Get per-class Dice scores
        dice_scores = self.dice_metric.aggregate()

        logging.info(f"Per-class Dice scores: {dice_scores}")
        logging.info(f"Mean Dice: {dice_scores.mean().item():.4f}")

        return np.concatenate(all_preds), np.concatenate(all_targets)
