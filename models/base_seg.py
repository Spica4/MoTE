"""
Base learner for 3D medical image segmentation with incremental learning.
"""

import copy
import logging
import numpy as np
import torch
from torch import nn
from utils.toolkit import tensor2numpy, accuracy
from monai.metrics import DiceMetric


class BaseSegLearner(object):
    """
    Base class for segmentation learners in incremental learning.

    Similar to BaseLearner but adapted for segmentation tasks.
    """

    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        # Segmentation-specific
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.use_init_ptm = args.get("use_init_ptm", False)
        self.moni_adam = args.get("moni_adam", False)

        # Evaluation settings
        self.topk = 5  # Not used for segmentation, kept for compatibility

    def after_task(self):
        pass

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        pass

    def _compute_accuracy(self, model, loader):
        """
        Compute accuracy for segmentation (using Dice score).
        """
        model.eval()
        self.dice_metric.reset()

        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                outputs = model.forward(inputs, test=True)

            # Get predictions
            preds = torch.argmax(outputs, dim=1, keepdim=True)

            # Compute Dice
            self.dice_metric(y_pred=preds, y=targets)

            # Pixel accuracy
            correct += (preds == targets).sum()
            total += targets.numel()

        dice_score = self.dice_metric.aggregate().mean().item()
        pixel_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

        logging.info(f"Pixel Accuracy: {pixel_acc}%, Mean Dice: {dice_score:.4f}")

        return dice_score * 100  # Return as percentage

    def eval_task(self):
        """
        Evaluate model on all seen tasks using TEST set (final evaluation only).

        This is called after training is complete to evaluate the final model
        performance on the held-out test set.

        Returns:
            cnn_accy: Dictionary with Dice scores
            nme_accy: None (not used for segmentation)
        """
        # Use test_loader for final evaluation (not used during training)
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate_dice(y_pred, y_true)

        return cnn_accy, None

    def _evaluate_dice(self, y_pred, y_true):
        """
        Evaluate Dice scores for different class groups.

        Args:
            y_pred: List of predicted segmentation masks (variable sizes)
            y_true: List of ground truth segmentation masks (variable sizes)

        Returns:
            Dictionary with Dice scores for different class groups
        """
        ret = {}

        # Compute overall Dice per-sample and average
        self.dice_metric.reset()
        for pred, target in zip(y_pred, y_true):
            self.dice_metric(
                y_pred=torch.from_numpy(pred),
                y=torch.from_numpy(target)
            )
        dice_scores = self.dice_metric.aggregate()
        mean_dice = dice_scores.mean().item()

        ret["top1"] = mean_dice * 100  # Overall Dice as percentage

        # Compute per-task Dice
        ret["grouped"] = {}
        for task_id in range(self._cur_task + 1):
            start_cls, end_cls = self._get_task_class_range(task_id)

            # Compute Dice for this task's classes
            task_dice = self._compute_task_dice_list(y_pred, y_true, start_cls, end_cls)
            ret["grouped"][f"{start_cls}-{end_cls-1}"] = task_dice

        # Top-5 not applicable for segmentation, use top1
        ret["top5"] = ret["top1"]

        return ret

    def _get_task_class_range(self, task_id):
        """Get class range for a task"""
        if task_id == 0:
            return 0, self.init_cls
        else:
            start = self.init_cls + (task_id - 1) * self.inc
            end = start + self.inc
            return start, end

    def _compute_task_dice_list(self, y_pred_list, y_true_list, start_cls, end_cls):
        """
        Compute Dice score for specific class range from lists of predictions.

        Args:
            y_pred_list: List of predicted masks (variable sizes)
            y_true_list: List of ground truth masks (variable sizes)
            start_cls: Start class index
            end_cls: End class index (exclusive)

        Returns:
            Mean Dice score for the class range
        """
        total_intersection = 0
        total_union = 0

        # Compute Dice across all samples
        for y_pred, y_true in zip(y_pred_list, y_true_list):
            # Create binary masks for task classes
            pred_mask = np.zeros_like(y_pred, dtype=bool)
            true_mask = np.zeros_like(y_true, dtype=bool)

            for cls in range(start_cls, end_cls):
                pred_mask |= (y_pred == cls)
                true_mask |= (y_true == cls)

            # Accumulate intersection and union
            total_intersection += np.sum(pred_mask & true_mask)
            total_union += np.sum(pred_mask) + np.sum(true_mask)

        if total_union == 0:
            return 0.0

        dice = (2.0 * total_intersection) / total_union
        return dice * 100  # Return as percentage

    def _compute_task_dice(self, y_pred, y_true, start_cls, end_cls):
        """
        Compute Dice score for specific class range.

        Args:
            y_pred: Predicted masks [N, 1, H, W, D]
            y_true: Ground truth masks [N, 1, H, W, D]
            start_cls: Start class index
            end_cls: End class index (exclusive)

        Returns:
            Mean Dice score for the class range
        """
        # Create binary masks for task classes
        pred_mask = np.zeros_like(y_pred, dtype=bool)
        true_mask = np.zeros_like(y_true, dtype=bool)

        for cls in range(start_cls, end_cls):
            pred_mask |= (y_pred == cls)
            true_mask |= (y_true == cls)

        # Compute Dice
        intersection = np.sum(pred_mask & true_mask)
        union = np.sum(pred_mask) + np.sum(true_mask)

        if union == 0:
            return 0.0

        dice = (2.0 * intersection) / union
        return dice * 100  # Return as percentage

    def _eval_cnn(self, loader):
        """
        Evaluate CNN model.

        Must be implemented by subclass.
        """
        raise NotImplementedError

    def _eval_nme(self, loader, class_means):
        """
        NME evaluation not used for segmentation.
        """
        return None

    @property
    def samples_old_class(self):
        return 0
