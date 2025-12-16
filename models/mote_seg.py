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
import os
import nibabel as nib

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
        # Exclude background from Dice loss to handle class imbalance
        # This focuses learning on foreground classes
        self.criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            reduction="mean",
            lambda_dice=1.0,
            lambda_ce=1.0,
        )

        # Metrics for evaluation
        # Include background in metric to see full segmentation quality
        self.dice_metric = DiceMetric(
            include_background=False,
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

        # Validation dataset (used during training)
        self.val_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="val", mode="test"
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Batch size 1 for sliding window inference
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Test dataset (used only for final evaluation)
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

        # Use val_loader for validation during training
        self._train(self.train_loader, self.val_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Freeze and add adapter to list
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def _train(self, train_loader, val_loader):
        """
        Main training loop.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (used during training)
        """
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

        self._init_train(train_loader, val_loader, optimizer, scheduler, epochs)

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
        elif self.args["scheduler"] == 'warmup_cosine':
            # Warmup + Cosine Annealing scheduler
            warmup_epochs = self.args.get("warmup_epochs", max(1, int(epochs * 0.1)))  # Default: 10% of total epochs

            # Linear warmup scheduler
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=0.01,  # Start from 1% of initial lr
                end_factor=1.0,
                total_iters=warmup_epochs
            )

            # Cosine annealing scheduler after warmup
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=self.min_lr
            )

            # Combine warmup and cosine schedulers
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )

            logging.info(f"Using warmup_cosine scheduler: warmup_epochs={warmup_epochs}, total_epochs={epochs}")
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

    def _init_train(self, train_loader, val_loader, optimizer, scheduler, epochs):
        """
        Training loop for current task.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            epochs: Number of training epochs
        """
        prog_bar = tqdm(range(epochs))

        # Track best model
        best_dice = 0.0
        best_model_state = None

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

            # Validation (using validation set)
            if (epoch + 1) % self.args.get("val_interval", 5) == 0:
                mean_dice = self._validate(val_loader)

                # Save best model
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_model_state = copy.deepcopy(self._network.state_dict())
                    logging.info(f"New best model! Val Dice: {best_dice:.4f}")

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

        # Load best model after training
        if best_model_state is not None:
            self._network.load_state_dict(best_model_state)
            logging.info(f"Loaded best model with Val Dice: {best_dice:.4f}")
        else:
            logging.warning("No validation performed, using final epoch model")

    def _validate(self, val_loader):
        """
        Validate model on validation set.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            mean_dice: Mean Dice score across validation samples
        """
        self._network.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for batch_idx, (_, inputs, targets) in enumerate(val_loader):
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
                outputs_pred = torch.argmax(outputs, dim=1, keepdim=True)

                self.dice_metric(y_pred=outputs_pred, y=targets)

        # Get mean Dice score
        mean_dice = self.dice_metric.aggregate().item()

        return mean_dice

    def _eval_cnn(self, loader, save_predictions=False):
        """
        Evaluate model using segmentation metrics.

        Args:
            loader: DataLoader for test data
            save_predictions: If True, save predictions as nii.gz files

        Returns predictions and ground truth for Dice computation.
        """
        self._network.eval()
        self.dice_metric.reset()

        all_preds = []
        all_targets = []

        # Create predictions directory if saving
        if save_predictions:
            pred_dir = os.path.join("predictions", f"task_{self._cur_task}")
            os.makedirs(pred_dir, exist_ok=True)
            logging.info(f"Saving predictions to {pred_dir}")

        with torch.no_grad():
            for batch_idx, (idx, inputs, targets) in enumerate(loader):
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

                # Convert to numpy for storage
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()

                all_preds.append(preds_np)
                all_targets.append(targets_np)

                # Save prediction as nii.gz if requested
                if save_predictions:
                    # Get the original image path from the dataset
                    data_dict = loader.dataset.data_dicts[idx.item() if torch.is_tensor(idx) else idx]
                    original_img_path = data_dict["image"]

                    # Load original image to get affine and header
                    original_img = nib.load(original_img_path)
                    affine = original_img.affine
                    header = original_img.header

                    # Create nifti image from prediction
                    # Remove batch and channel dimensions: [1, 1, H, W, D] -> [H, W, D]
                    pred_data = preds_np[0, 0, :, :, :]

                    # Create nifti image
                    pred_nifti = nib.Nifti1Image(pred_data.astype(np.int16), affine, header)

                    # Save prediction
                    pred_filename = os.path.basename(original_img_path).replace(".nii.gz", "_pred.nii.gz")
                    pred_path = os.path.join(pred_dir, pred_filename)
                    nib.save(pred_nifti, pred_path)

                    logging.info(f"Saved prediction [{batch_idx + 1}/{len(loader)}]: {pred_path}")

        # Get per-class Dice scores
        dice_scores = self.dice_metric.aggregate()

        logging.info(f"Per-class Dice scores: {dice_scores}")
        logging.info(f"Mean Dice: {dice_scores.mean().item():.4f}")

        # Return lists instead of concatenated arrays
        # Medical images have varying sizes, so we can't concatenate them
        return all_preds, all_targets
