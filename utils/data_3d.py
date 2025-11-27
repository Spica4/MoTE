"""
3D Medical Image Dataset utilities for nii.gz files
Supports incremental learning for medical image segmentation tasks
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import nibabel as nib
from monai import transforms
from monai.data import Dataset, CacheDataset
from utils.data import iData


def build_transform_3d(is_train: bool, args: dict, roi_size: Tuple[int, int, int] = (96, 96, 96)):
    """
    Build 3D transforms for medical images using MONAI

    Args:
        is_train: Whether this is for training (with augmentation)
        args: Configuration arguments
        roi_size: Region of interest size for cropping/padding

    Returns:
        List of MONAI transforms
    """
    if is_train:
        transform = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            transforms.RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    else:
        transform = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]

    return transform


class MedicalImage3D(iData):
    """
    Base class for 3D medical image segmentation datasets

    This class handles NIfTI (.nii.gz) format medical images for
    incremental learning segmentation tasks.
    """
    use_path = True
    is_segmentation = True  # Flag to indicate this is a segmentation task

    def __init__(self, args: dict, roi_size: Tuple[int, int, int] = (96, 96, 96)):
        """
        Args:
            args: Configuration dictionary
            roi_size: Region of interest size for patches
        """
        super().__init__()
        self.args = args
        self.roi_size = roi_size

        self.train_trsf = build_transform_3d(True, args, roi_size)
        self.test_trsf = build_transform_3d(False, args, roi_size)
        self.common_trsf = []

        # Will be set by subclasses
        self.class_order = None
        self.num_classes = None

    def load_nifti_paths(self, data_dir: str, split: str = "train") -> Tuple[List[str], List[str]]:
        """
        Load paths to NIfTI image and label files

        Args:
            data_dir: Root directory containing the dataset
            split: 'train' or 'test'

        Returns:
            Tuple of (image_paths, label_paths)
        """
        data_path = Path(data_dir) / split

        # Assuming directory structure:
        # data_dir/train/images/*.nii.gz
        # data_dir/train/labels/*.nii.gz
        image_dir = data_path / "images"
        label_dir = data_path / "labels"

        if not image_dir.exists() or not label_dir.exists():
            raise ValueError(f"Image or label directory not found in {data_path}")

        image_files = sorted(list(image_dir.glob("*.nii.gz")))
        label_files = sorted(list(label_dir.glob("*.nii.gz")))

        if len(image_files) != len(label_files):
            raise ValueError(
                f"Number of images ({len(image_files)}) != number of labels ({len(label_files)})"
            )

        return [str(f) for f in image_files], [str(f) for f in label_files]

    def create_data_dicts(self, image_paths: List[str], label_paths: List[str]) -> List[Dict]:
        """
        Create data dictionaries for MONAI Dataset

        Args:
            image_paths: List of paths to image files
            label_paths: List of paths to label files

        Returns:
            List of dictionaries with 'image' and 'label' keys
        """
        data_dicts = []
        for img_path, lbl_path in zip(image_paths, label_paths):
            data_dicts.append({
                "image": img_path,
                "label": lbl_path,
            })
        return data_dicts


class BTSDataset(MedicalImage3D):
    """
    Brain Tumor Segmentation (BraTS) dataset

    Supports incremental learning on different tumor types or regions
    """

    def __init__(self, args: dict, roi_size: Tuple[int, int, int] = (128, 128, 128)):
        super().__init__(args, roi_size)
        # BraTS typically has 4 classes: background, necrotic, edema, enhancing
        self.num_classes = 4
        self.class_order = np.arange(self.num_classes).tolist()

    def download_data(self):
        """
        Load BraTS dataset from specified directory

        Expected structure:
        /datasets/brats/train/images/*.nii.gz
        /datasets/brats/train/labels/*.nii.gz
        /datasets/brats/test/images/*.nii.gz
        /datasets/brats/test/labels/*.nii.gz
        """
        data_dir = self.args.get("data_dir", "/datasets/brats")

        train_images, train_labels = self.load_nifti_paths(data_dir, "train")
        test_images, test_labels = self.load_nifti_paths(data_dir, "test")

        # Store as data dictionaries for MONAI
        self.train_data = self.create_data_dicts(train_images, train_labels)
        self.test_data = self.create_data_dicts(test_images, test_labels)

        # For segmentation, we don't have simple targets like classification
        # Instead, we store the paths
        self.train_targets = np.arange(len(self.train_data))
        self.test_targets = np.arange(len(self.test_data))


class AbdominalDataset(MedicalImage3D):
    """
    Abdominal multi-organ segmentation dataset

    Supports incremental learning on different organs
    Common organs: liver, kidney, spleen, pancreas, etc.
    """

    def __init__(self, args: dict, roi_size: Tuple[int, int, int] = (96, 96, 96)):
        super().__init__(args, roi_size)
        # Number of organs + background
        self.num_classes = args.get("num_classes", 14)  # e.g., 13 organs + background
        self.class_order = np.arange(self.num_classes).tolist()

    def download_data(self):
        """
        Load abdominal dataset from specified directory
        """
        data_dir = self.args.get("data_dir", "/datasets/abdomen")

        train_images, train_labels = self.load_nifti_paths(data_dir, "train")
        test_images, test_labels = self.load_nifti_paths(data_dir, "test")

        self.train_data = self.create_data_dicts(train_images, train_labels)
        self.test_data = self.create_data_dicts(test_images, test_labels)

        self.train_targets = np.arange(len(self.train_data))
        self.test_targets = np.arange(len(self.test_data))


class CustomMedical3D(MedicalImage3D):
    """
    Generic 3D medical image dataset

    Can be used for any custom NIfTI segmentation dataset
    """

    def __init__(self, args: dict, roi_size: Tuple[int, int, int] = (96, 96, 96)):
        super().__init__(args, roi_size)
        self.num_classes = args.get("num_classes", 2)  # Default: binary segmentation
        self.class_order = np.arange(self.num_classes).tolist()

    def download_data(self):
        """
        Load custom 3D medical dataset from specified directory
        """
        data_dir = self.args.get("data_dir")
        if data_dir is None:
            raise ValueError("data_dir must be specified in args for CustomMedical3D")

        train_images, train_labels = self.load_nifti_paths(data_dir, "train")
        test_images, test_labels = self.load_nifti_paths(data_dir, "test")

        self.train_data = self.create_data_dicts(train_images, train_labels)
        self.test_data = self.create_data_dicts(test_images, test_labels)

        self.train_targets = np.arange(len(self.train_data))
        self.test_targets = np.arange(len(self.test_data))
