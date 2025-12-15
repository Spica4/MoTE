"""
3D Medical Image Dataset for Incremental Learning with Swin UNETR and MoTE

Supports multi-stage learning with different datasets:
- Stage 1: Kaken dataset (6 classes from 17 original labels)
- Stage 2: amos22 dataset (6 additional classes from 15 original labels)
"""

import os
import glob
import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    MapLabelValued,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    Spacingd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    Lambdad,
    ToTensord,
)


class Medical3DData(object):
    """Base class for 3D medical image datasets"""

    use_path = True
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    def __init__(self, args):
        self.args = args


class KakenDataset(Medical3DData):
    """
    Kaken dataset for torso organ segmentation.

    Original labels: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    Fixed labels:    [0,0,0,1,2,3,4,5,6,0,0,0,0,0,0,0,0]

    Only uses 6 labels (1-6) in stage 1 training.
    """

    use_path = True

    # Original and target label mapping
    orig_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    fixed_labels = [0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0]

    # Class order for incremental learning (using labels 1-6, excluding background 0)
    class_order = np.arange(1, 7).tolist()  # [1, 2, 3, 4, 5, 6]

    def __init__(self, args):
        super().__init__(args)
        self.data_dir = "/deeparea/sokabe/Dataset/Kaken_set_A/torso/1.5mm"
        self.image_dir = os.path.join(self.data_dir, "image")
        self.label_dir = os.path.join(self.data_dir, "label")

        # Setup transforms
        self.train_trsf = self._get_train_transforms()
        self.test_trsf = self._get_val_transforms()
        self.common_trsf = []

    def _get_train_transforms(self):
        """Get training transforms for Kaken dataset"""
        # Get img_size from args, default to [96, 96, 96]
        img_size = tuple(self.args.get("img_size", [96, 96, 96]))

        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            MapLabelValued(
                keys=["label"],
                orig_labels=self.orig_labels,
                target_labels=self.fixed_labels,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            SpatialPadd(keys=["image", "label"], spatial_size=img_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=img_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            Lambdad(
                keys=["label"],
                func=lambda x: torch.where(x > 6, torch.tensor(0, dtype=x.dtype, device=x.device), x)
            ),
            ToTensord(keys=["image", "label"]),
        ])

    def _get_val_transforms(self):
        """Get validation transforms for Kaken dataset"""
        # Get img_size from args, default to [96, 96, 96]
        img_size = tuple(self.args.get("img_size", [96, 96, 96]))

        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            MapLabelValued(
                keys=["label"],
                orig_labels=self.orig_labels,
                target_labels=self.fixed_labels,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            SpatialPadd(keys=["image", "label"], spatial_size=img_size),
            Lambdad(
                keys=["label"],
                func=lambda x: torch.where(x > 6, torch.tensor(0, dtype=x.dtype, device=x.device), x)
            ),
            ToTensord(keys=["image", "label"]),
        ])

    def download_data(self):
        """Load Kaken dataset file paths"""
        # Get all image and label file paths
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))
        label_files = sorted(glob.glob(os.path.join(self.label_dir, "*.nii.gz")))

        if len(image_files) == 0:
            raise ValueError(f"No image files found in {self.image_dir}")
        if len(label_files) == 0:
            raise ValueError(f"No label files found in {self.label_dir}")

        print(f"Found {len(image_files)} images and {len(label_files)} labels in Kaken dataset")

        # Create data dictionaries
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(image_files, label_files)
        ]

        # Split into train/val/test according to specification
        # Total: 89 samples
        # Train: 0-53 (54 samples)
        # Val: 54-71 (18 samples)
        # Test: 72-88 (17 samples)
        train_files = data_dicts[0:54]
        val_files = data_dicts[54:72]
        test_files = data_dicts[72:89]

        # Store as numpy arrays of file paths
        self.train_data = np.array([d["image"] for d in train_files])
        self.train_targets = np.zeros(len(train_files), dtype=np.int64)  # Dummy targets for compatibility

        self.test_data = np.array([d["image"] for d in test_files])
        self.test_targets = np.zeros(len(test_files), dtype=np.int64)

        # Store validation data
        self.val_data = np.array([d["image"] for d in val_files])
        self.val_targets = np.zeros(len(val_files), dtype=np.int64)

        # Store full data dicts for later use
        self.train_data_dicts = train_files
        self.val_data_dicts = val_files
        self.test_data_dicts = test_files

        print(f"Kaken split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


class Amos22Dataset(Medical3DData):
    """
    AMOS22 dataset for abdominal organ segmentation.

    Original labels: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    Fixed labels:    [0,7,8,0,0,0,0,0,9,10,0,0,0,11,12]

    Uses labels 7-12 in stage 2 training (6 classes).
    Note: These are different from stage 1's labels 1-6.
    """

    use_path = True

    # Original and target label mapping
    orig_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    fixed_labels = [0, 7, 8, 0, 0, 0, 0, 0, 9, 10, 0, 0, 0, 11, 12]

    # Class order for incremental learning (using labels 7-12)
    class_order = np.arange(7, 13).tolist()  # [7, 8, 9, 10, 11, 12]

    def __init__(self, args):
        super().__init__(args)
        self.data_dir = "/deeparea/sokabe/Dataset/amos22/amos22"
        self.image_dir = os.path.join(self.data_dir, "imagesTr")
        self.label_dir = os.path.join(self.data_dir, "labelsTr")

        # Setup transforms
        self.train_trsf = self._get_train_transforms()
        self.test_trsf = self._get_val_transforms()
        self.common_trsf = []

    def _get_train_transforms(self):
        """Get training transforms for AMOS22 dataset"""
        # Get img_size from args, default to [96, 96, 96]
        img_size = tuple(self.args.get("img_size", [96, 96, 96]))

        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            MapLabelValued(
                keys=["label"],
                orig_labels=self.orig_labels,
                target_labels=self.fixed_labels,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            SpatialPadd(keys=["image", "label"], spatial_size=img_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=img_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            # Keep only labels 7-12, set others to 0
            Lambdad(
                keys=["label"],
                func=lambda x: torch.where(
                    (x >= 7) & (x <= 12),
                    x,
                    torch.tensor(0, dtype=x.dtype, device=x.device)
                )
            ),
            ToTensord(keys=["image", "label"]),
        ])

    def _get_val_transforms(self):
        """Get validation transforms for AMOS22 dataset"""
        # Get img_size from args, default to [96, 96, 96]
        img_size = tuple(self.args.get("img_size", [96, 96, 96]))

        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            MapLabelValued(
                keys=["label"],
                orig_labels=self.orig_labels,
                target_labels=self.fixed_labels,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            SpatialPadd(keys=["image", "label"], spatial_size=img_size),
            Lambdad(
                keys=["label"],
                func=lambda x: torch.where(
                    (x >= 7) & (x <= 12),
                    x,
                    torch.tensor(0, dtype=x.dtype, device=x.device)
                )
            ),
            ToTensord(keys=["image", "label"]),
        ])

    def download_data(self):
        """Load AMOS22 dataset file paths"""
        # Get all image and label file paths
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))
        label_files = sorted(glob.glob(os.path.join(self.label_dir, "*.nii.gz")))

        if len(image_files) == 0:
            raise ValueError(f"No image files found in {self.image_dir}")
        if len(label_files) == 0:
            raise ValueError(f"No label files found in {self.label_dir}")

        print(f"Found {len(image_files)} images and {len(label_files)} labels in AMOS22 dataset")

        # Create data dictionaries
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(image_files, label_files)
        ]

        # Split into train/val/test according to specification
        # Total: 200 samples
        # Train: 0-119 (120 samples)
        # Val: 120-159 (40 samples)
        # Test: 160-199 (40 samples)
        train_files = data_dicts[0:120]
        val_files = data_dicts[120:160]
        test_files = data_dicts[160:200]

        # Store as numpy arrays of file paths
        self.train_data = np.array([d["image"] for d in train_files])
        self.train_targets = np.zeros(len(train_files), dtype=np.int64)  # Dummy targets

        self.test_data = np.array([d["image"] for d in test_files])
        self.test_targets = np.zeros(len(test_files), dtype=np.int64)

        # Store validation data
        self.val_data = np.array([d["image"] for d in val_files])
        self.val_targets = np.zeros(len(val_files), dtype=np.int64)

        # Store full data dicts for later use
        self.train_data_dicts = train_files
        self.val_data_dicts = val_files
        self.test_data_dicts = test_files

        print(f"AMOS22 split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


class TwoStageDataset(Medical3DData):
    """
    Two-stage dataset that combines Kaken (stage 1) and AMOS22 (stage 2).

    This dataset handles the incremental learning scenario where:
    - Task 0: Kaken dataset with labels 1-6
    - Task 1: AMOS22 dataset with labels 7-12
    """

    use_path = True
    class_order = np.arange(1, 13).tolist()  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def __init__(self, args):
        super().__init__(args)

        # Initialize both datasets
        self.kaken = KakenDataset(args)
        self.amos22 = Amos22Dataset(args)

        # Default to using combined class order
        self.current_stage = 0  # 0 for Kaken, 1 for AMOS22

        # Set initial transforms to Kaken's transforms
        self.train_trsf = self.kaken.train_trsf
        self.test_trsf = self.kaken.test_trsf

    def download_data(self):
        """Load both datasets"""
        self.kaken.download_data()
        self.amos22.download_data()

        # Initially use Kaken data
        self.train_data = self.kaken.train_data
        self.train_targets = self.kaken.train_targets
        self.test_data = self.kaken.test_data
        self.test_targets = self.kaken.test_targets
        self.val_data = self.kaken.val_data
        self.val_targets = self.kaken.val_targets

        # Store data dicts
        self.train_data_dicts = self.kaken.train_data_dicts
        self.val_data_dicts = self.kaken.val_data_dicts
        self.test_data_dicts = self.kaken.test_data_dicts

        print("Two-stage dataset initialized with Kaken (stage 1)")

    def switch_to_stage2(self):
        """Switch to AMOS22 dataset for stage 2 training"""
        self.current_stage = 1
        self.train_data = self.amos22.train_data
        self.train_targets = self.amos22.train_targets
        self.test_data = self.amos22.test_data
        self.test_targets = self.amos22.test_targets
        self.val_data = self.amos22.val_data
        self.val_targets = self.amos22.val_targets

        self.train_data_dicts = self.amos22.train_data_dicts
        self.val_data_dicts = self.amos22.val_data_dicts
        self.test_data_dicts = self.amos22.test_data_dicts

        # Update transforms to AMOS22's transforms
        self.train_trsf = self.amos22.train_trsf
        self.test_trsf = self.amos22.test_trsf

        print("Switched to AMOS22 (stage 2)")

    def get_current_transforms(self, mode="train"):
        """Get transforms for current stage and mode"""
        dataset = self.kaken if self.current_stage == 0 else self.amos22

        if mode == "train":
            return dataset.train_trsf
        else:
            return dataset.test_trsf
