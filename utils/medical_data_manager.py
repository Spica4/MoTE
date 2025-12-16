"""
Data Manager for 3D Medical Image Segmentation with Incremental Learning

Handles two-stage learning:
- Stage 1: Kaken dataset (classes 1-6)
- Stage 2: AMOS22 dataset (classes 7-12)
"""

import logging
import numpy as np
from torch.utils.data import Dataset
from monai.data import Dataset as MonaiDataset
from utils.medical_data import KakenDataset, Amos22Dataset, TwoStageDataset


class MedicalDataManager(object):
    """
    Data manager for 3D medical image segmentation with incremental learning.

    Differs from standard DataManager in that:
    - Uses 3D NIfTI images instead of 2D images
    - Uses MONAI transforms instead of torchvision transforms
    - Handles segmentation labels (dense predictions) instead of class labels
    - Supports two-stage learning with different datasets
    """

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)

        # For medical segmentation, we have predefined class splits
        # Task 0: classes 1-6 (Kaken)
        # Task 1: classes 7-12 (AMOS22)
        if dataset_name.lower() == "twostage":
            # Two tasks: Kaken (6 classes) and AMOS22 (6 classes)
            self._increments = [6, 6]  # Task 0: 6 classes, Task 1: 6 classes
        else:
            # Single dataset
            assert init_cls <= len(self._class_order), "No enough classes."
            self._increments = [init_cls]
            while sum(self._increments) + increment < len(self._class_order):
                self._increments.append(increment)
            offset = len(self._class_order) - sum(self._increments)
            if offset > 0:
                self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def prepare_task(self, task_id):
        """
        Prepare data manager for a specific task.

        For two-stage learning, this switches to stage 2 data when task_id == 1.
        """
        if self._current_task != task_id:
            self._current_task = task_id

            # For two-stage dataset, switch to stage 2 when task_id == 1
            if self.dataset_name.lower() == "twostage" and task_id == 1:
                logging.info("Switching to Stage 2 (AMOS22 dataset)")
                self._idata.switch_to_stage2()

                # Update data references
                self._train_data_dicts = self._idata.train_data_dicts
                self._test_data_dicts = self._idata.test_data_dicts
                if hasattr(self._idata, 'val_data_dicts'):
                    self._val_data_dicts = self._idata.val_data_dicts

                # Update transforms
                self._train_trsf = self._idata.train_trsf
                self._test_trsf = self._idata.test_trsf

                logging.info("Stage 2 preparation complete")

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        """
        Get dataset for specified indices and mode.

        For medical segmentation:
        - indices: class indices to include (e.g., [1, 2, 3, 4, 5, 6] for task 0)
        - source: 'train', 'val', or 'test'
        - mode: 'train' or 'test' (affects transforms)
        """
        if source == "train":
            data_dicts = self._train_data_dicts
        elif source == "val":
            data_dicts = self._val_data_dicts
        elif source == "test":
            data_dicts = self._test_data_dicts
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # Get transforms for current mode
        if mode == "train":
            trsf = self._train_trsf
        elif mode == "test":
            trsf = self._test_trsf
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # For medical images, we use all data (no class filtering at data level)
        # Class filtering happens through label processing in transforms
        if ret_data:
            return data_dicts, None, MedicalDummyDataset(data_dicts, trsf)
        else:
            return MedicalDummyDataset(data_dicts, trsf)

    def _setup_data(self, dataset_name, shuffle, seed):
        """Setup data based on dataset name"""
        idata = _get_medical_idata(dataset_name, self.args)
        idata.download_data()

        # Data dictionaries (containing image/label paths)
        self._train_data_dicts = idata.train_data_dicts
        self._test_data_dicts = idata.test_data_dicts
        if hasattr(idata, 'val_data_dicts'):
            self._val_data_dicts = idata.val_data_dicts

        # Transforms
        self._train_trsf = idata.train_trsf if hasattr(idata, 'train_trsf') else None
        self._test_trsf = idata.test_trsf if hasattr(idata, 'test_trsf') else None

        # Class order
        self._class_order = idata.class_order
        logging.info(f"Class order: {self._class_order}")

        # Store dataset reference for two-stage learning
        self._idata = idata
        self._current_task = -1


class MedicalDummyDataset(Dataset):
    """
    Dummy dataset for medical images that applies MONAI transforms.

    Unlike standard DummyDataset, this:
    - Takes data dictionaries with 'image' and 'label' keys
    - Applies MONAI transforms that handle both image and label
    - Returns 3D volumes instead of 2D images
    """

    def __init__(self, data_dicts, transforms=None):
        self.data_dicts = data_dicts
        self.transforms = transforms

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data = self.data_dicts[idx].copy()

        # Apply MONAI transforms
        if self.transforms is not None:
            data = self.transforms(data)

        # Handle list output from transforms (e.g., RandCropByPosNegLabeld with num_samples > 1)
        # MONAI sometimes returns a list even with num_samples=1
        if isinstance(data, list):
            data = data[0]  # Take the first sample

        # Extract image and label
        image = data["image"]
        label = data["label"]

        # Return idx, image, label (same format as original DummyDataset)
        return idx, image, label


def _get_medical_idata(dataset_name, args):
    """Get medical image dataset instance"""
    name = dataset_name.lower()

    if name == "kaken":
        return KakenDataset(args)
    elif name == "amos22":
        return Amos22Dataset(args)
    elif name == "twostage":
        return TwoStageDataset(args)
    else:
        raise NotImplementedError("Unknown medical dataset {}.".format(dataset_name))
