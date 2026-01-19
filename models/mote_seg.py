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
import csv

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

    def _compute_per_organ_dice(self, pred, target, num_classes=13):
        """
        Compute Dice score for each organ class.

        Args:
            pred: Prediction array [1, H, W, D]
            target: Ground truth array [1, H, W, D]
            num_classes: Number of classes (including background)

        Returns:
            List of Dice scores for each class (0-12)
        """
        dice_scores = []

        for class_id in range(num_classes):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)

            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask) + np.sum(target_mask)

            if union == 0:
                # If both pred and target are empty for this class, Dice = NaN
                dice = np.nan
            else:
                dice = (2.0 * intersection) / union

            dice_scores.append(dice)

        return dice_scores

    def eval_task(self):
        """
        Evaluate model on task-specific test sets.

        For Task 1, this evaluates:
        - Task 0 classes (1-6) on Kaken test set
        - Task 1 classes (7-12) on AMOS22 test set

        Returns:
            cnn_accy: Dictionary with Dice scores
            nme_accy: None (not used for segmentation)
        """
        if self._cur_task == 0:
            # For Task 0, use the base implementation (evaluate on Kaken test set)
            y_pred, y_true = self._eval_cnn(self.test_loader, save_predictions=True)
            cnn_accy = self._evaluate_dice(y_pred, y_true)
        else:
            # For Task 1+, evaluate each task on its respective test set
            logging.info("=" * 60)
            logging.info("Task-specific evaluation:")
            logging.info("  - Task 0 classes (1-6) on Kaken test set")
            logging.info("  - Task 1 classes (7-12) on AMOS22 test set")
            logging.info("=" * 60)

            # Evaluate Task 0 classes (1-6) on Kaken test set
            logging.info("\nEvaluating Task 0 classes (1-6) on Kaken test set...")
            y_pred_task0, y_true_task0 = self._eval_task_specific(
                task_id=0,
                dataset_name="kaken",
                save_predictions=True
            )

            # Evaluate Task 1 classes (7-12) on AMOS22 test set
            logging.info("\nEvaluating Task 1 classes (7-12) on AMOS22 test set...")
            y_pred_task1, y_true_task1 = self._eval_task_specific(
                task_id=1,
                dataset_name="amos22",
                save_predictions=True
            )

            # Compute Dice scores for each task
            task0_dice = self._compute_task_dice_list(y_pred_task0, y_true_task0, 1, 7)
            task1_dice = self._compute_task_dice_list(y_pred_task1, y_true_task1, 7, 13)

            # Combine results
            cnn_accy = {
                "top1": (task0_dice + task1_dice) / 2.0,  # Average of both tasks
                "top5": (task0_dice + task1_dice) / 2.0,
                "grouped": {
                    "1-6": task0_dice,
                    "7-12": task1_dice
                }
            }

            logging.info("\n" + "=" * 60)
            logging.info("Task-specific evaluation results:")
            logging.info(f"  Task 0 (classes 1-6) on Kaken: {task0_dice:.2f}%")
            logging.info(f"  Task 1 (classes 7-12) on AMOS22: {task1_dice:.2f}%")
            logging.info(f"  Overall average: {cnn_accy['top1']:.2f}%")
            logging.info("=" * 60)

        return cnn_accy, None

    def _eval_task_specific(self, task_id, dataset_name, save_predictions=False):
        """
        Evaluate model on a specific task's test set.

        Args:
            task_id: Task ID (0 for Kaken, 1 for AMOS22)
            dataset_name: 'kaken' or 'amos22'
            save_predictions: Whether to save predictions as nii.gz

        Returns:
            y_pred: List of predicted masks
            y_true: List of ground truth masks
        """
        # Get the appropriate dataset from data_manager
        if hasattr(self.data_manager._idata, dataset_name):
            dataset_obj = getattr(self.data_manager._idata, dataset_name)
            test_data_dicts = dataset_obj.test_data_dicts
            test_trsf = dataset_obj.test_trsf
        else:
            raise ValueError(f"Dataset {dataset_name} not found in data_manager")

        # Create test dataset for this task
        from utils.medical_data_manager import MedicalDummyDataset
        test_dataset = MedicalDummyDataset(test_data_dicts, test_trsf)

        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        logging.info(f"Evaluating on {dataset_name} test set ({len(test_dataset)} samples)")

        # Evaluate using the specific loader
        y_pred, y_true = self._eval_cnn_task_specific(
            test_loader,
            task_id=task_id,
            dataset_name=dataset_name,
            save_predictions=save_predictions
        )

        return y_pred, y_true

    def _eval_cnn_task_specific(self, loader, task_id, dataset_name, save_predictions=False):
        """
        Evaluate model on a specific task's test set.

        Args:
            loader: DataLoader for test data
            task_id: Task ID for naming predictions
            dataset_name: Dataset name for prediction directory
            save_predictions: Whether to save predictions as nii.gz

        Returns:
            all_preds: List of predicted masks
            all_targets: List of ground truth masks
        """
        self._network.eval()
        self.dice_metric.reset()

        all_preds = []
        all_targets = []

        # For CSV export
        csv_results = []

        # Organ names in Japanese (classes 1-12)
        organ_names = [
            "背景",  # Class 0 (background)
            "大動脈",  # Class 1
            "食道",  # Class 2
            "肝臓",  # Class 3
            "胆嚢",  # Class 4
            "胃",  # Class 5
            "脾臓",  # Class 6
            "右腎臓",  # Class 7
            "左腎臓",  # Class 8
            "下大動脈",  # Class 9
            "膵臓",  # Class 10
            "膀胱",  # Class 11
            "子宮",  # Class 12
        ]

        # Create predictions directory if saving
        if save_predictions:
            pred_dir = os.path.join("predictions", f"task_{self._cur_task}", f"task{task_id}_{dataset_name}")
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

                # Get the original image path from the dataset
                data_dict = loader.dataset.data_dicts[idx.item() if torch.is_tensor(idx) else idx]
                original_img_path = data_dict["image"]
                filename = os.path.basename(original_img_path)

                # Compute per-organ Dice scores for this image
                per_organ_dice = self._compute_per_organ_dice(preds_np[0], targets_np[0])

                # Store results for CSV export
                result_row = {"filename": filename}
                for class_id in range(1, 13):  # Classes 1-12 only
                    organ_name = organ_names[class_id]
                    dice_value = per_organ_dice[class_id]
                    result_row[organ_name] = dice_value

                # Compute average Dice across organs (excluding background and NaN values)
                valid_dice_scores = [per_organ_dice[i] for i in range(1, 13) if not np.isnan(per_organ_dice[i])]
                if valid_dice_scores:
                    avg_dice = np.mean(valid_dice_scores)
                else:
                    avg_dice = 0.0
                result_row["平均"] = avg_dice

                csv_results.append(result_row)

                # Save prediction as nii.gz if requested
                if save_predictions:
                    # Load original image to get size, affine and header
                    original_img = nib.load(original_img_path)
                    original_shape = original_img.shape
                    affine = original_img.affine
                    header = original_img.header

                    # Get prediction data: [1, 1, H, W, D] -> [H, W, D]
                    pred_data = preds_np[0, 0, :, :, :]

                    # Resize prediction to original image size
                    if pred_data.shape != original_shape:
                        from scipy.ndimage import zoom

                        # Calculate zoom factors for each dimension
                        zoom_factors = [
                            original_shape[0] / pred_data.shape[0],
                            original_shape[1] / pred_data.shape[1],
                            original_shape[2] / pred_data.shape[2],
                        ]

                        # Use nearest neighbor interpolation for segmentation labels
                        pred_data_resized = zoom(pred_data, zoom_factors, order=0)

                        logging.info(f"Resized prediction from {pred_data.shape} to {pred_data_resized.shape} (original: {original_shape})")
                        pred_data = pred_data_resized

                    # Create nifti image with original size and affine
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

        # Save results to CSV if we have predictions
        if save_predictions and csv_results:
            csv_dir = os.path.join("predictions", f"task_{self._cur_task}", f"task{task_id}_{dataset_name}")
            csv_path = os.path.join(csv_dir, f"evaluation_results_task{task_id}_{dataset_name}.csv")

            # Write CSV file
            fieldnames = ["filename"] + [organ_names[i] for i in range(1, 13)] + ["平均"]
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_results)

            logging.info(f"Saved evaluation results to {csv_path}")

        return all_preds, all_targets

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

        # For CSV export
        csv_results = []  # List of dicts with filename and per-organ dice scores

        # Organ names in Japanese (classes 1-12)
        organ_names = [
            "背景",  # Class 0 (background)
            "大動脈",  # Class 1
            "食道",  # Class 2
            "肝臓",  # Class 3
            "胆嚢",  # Class 4
            "胃",  # Class 5
            "脾臓",  # Class 6
            "右腎臓",  # Class 7
            "左腎臓",  # Class 8
            "下大動脈",  # Class 9
            "膵臓",  # Class 10
            "膀胱",  # Class 11
            "子宮",  # Class 12
        ]

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

                # Get the original image path from the dataset
                data_dict = loader.dataset.data_dicts[idx.item() if torch.is_tensor(idx) else idx]
                original_img_path = data_dict["image"]
                filename = os.path.basename(original_img_path)

                # Compute per-organ Dice scores for this image
                per_organ_dice = self._compute_per_organ_dice(preds_np[0], targets_np[0])

                # Store results for CSV export
                result_row = {"filename": filename}
                for class_id in range(1, 13):  # Classes 1-12 only
                    organ_name = organ_names[class_id]
                    dice_value = per_organ_dice[class_id]
                    result_row[organ_name] = dice_value

                # Compute average Dice across organs (excluding background and NaN values)
                valid_dice_scores = [per_organ_dice[i] for i in range(1, 13) if not np.isnan(per_organ_dice[i])]
                if valid_dice_scores:
                    avg_dice = np.mean(valid_dice_scores)
                else:
                    avg_dice = 0.0
                result_row["平均"] = avg_dice

                csv_results.append(result_row)

                # Save prediction as nii.gz if requested
                if save_predictions:
                    # Load original image to get size, affine and header
                    original_img = nib.load(original_img_path)
                    original_shape = original_img.shape
                    affine = original_img.affine
                    header = original_img.header

                    # Get prediction data: [1, 1, H, W, D] -> [H, W, D]
                    pred_data = preds_np[0, 0, :, :, :]

                    # Resize prediction to original image size
                    if pred_data.shape != original_shape:
                        from scipy.ndimage import zoom

                        # Calculate zoom factors for each dimension
                        zoom_factors = [
                            original_shape[0] / pred_data.shape[0],
                            original_shape[1] / pred_data.shape[1],
                            original_shape[2] / pred_data.shape[2],
                        ]

                        # Use nearest neighbor interpolation for segmentation labels
                        pred_data_resized = zoom(pred_data, zoom_factors, order=0)

                        logging.info(f"Resized prediction from {pred_data.shape} to {pred_data_resized.shape} (original: {original_shape})")
                        pred_data = pred_data_resized

                    # Create nifti image with original size and affine
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

        # Save results to CSV if we have predictions
        if save_predictions and csv_results:
            csv_dir = os.path.join("predictions", f"task_{self._cur_task}")
            csv_path = os.path.join(csv_dir, f"evaluation_results_task_{self._cur_task}.csv")

            # Write CSV file
            fieldnames = ["filename"] + [organ_names[i] for i in range(1, 13)] + ["平均"]
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_results)

            logging.info(f"Saved evaluation results to {csv_path}")

        # Return lists instead of concatenated arrays
        # Medical images have varying sizes, so we can't concatenate them
        return all_preds, all_targets
