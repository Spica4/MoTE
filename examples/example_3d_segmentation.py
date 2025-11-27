"""
Example script for 3D medical image segmentation with MoTE

This script demonstrates how to use the MoTE framework for
incremental learning on 3D medical image segmentation tasks.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from trainer import train


def example_brats_segmentation():
    """
    Example: BraTS brain tumor segmentation with incremental learning

    Task 0: Learn background + necrotic core
    Task 1: Add edema
    Task 2: Add enhancing tumor
    """
    config = {
        "prefix": "BraTS Example",
        "dataset": "brats",
        "data_dir": "/datasets/brats",
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": False,
        "shuffle": False,
        "init_cls": 2,  # Start with background + necrotic
        "increment": 1,  # Add one class at a time
        "model_name": "mote_seg",
        "backbone_type": "swin_unetr_mote_base",
        "device": [0],  # GPU 0
        "seed": [1993],

        # Training parameters
        "init_epochs": 100,
        "init_lr": 0.0001,
        "later_epochs": 50,
        "later_lr": 0.00005,
        "batch_size": 2,
        "weight_decay": 1e-5,
        "min_lr": 1e-6,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "pretrained": True,

        # Adapter parameters
        "ffn_num": 64,
        "d_model": 768,
        "alpha": 0.005,
        "use_init_ptm": False,
        "beta": 0,

        # 3D segmentation parameters
        "roi_size": [128, 128, 128],
        "sw_batch_size": 4,
        "overlap": 0.5,
        "use_dice_ce": True,
        "val_interval": 5,

        # Other parameters
        "moni_adam": False,
        "adapter_num": -1,
        "print_forget": True,

        # Dataset-specific
        "num_classes": 4,
        "in_channels": 4,  # BraTS has 4 modalities
        "task_type": "segmentation",

        # Computed
        "nb_classes": 4,
        "nb_tasks": 3,  # Will be computed by data manager
    }

    print("=" * 80)
    print("Example: BraTS Brain Tumor Segmentation with Incremental Learning")
    print("=" * 80)
    print(f"Initial classes: {config['init_cls']}")
    print(f"Increment: {config['increment']}")
    print(f"Total classes: {config['num_classes']}")
    print(f"Backbone: {config['backbone_type']}")
    print(f"ROI size: {config['roi_size']}")
    print("=" * 80)

    # Run training
    train(config)


def example_abdomen_segmentation():
    """
    Example: Abdominal organ segmentation with incremental learning

    Task 0: Learn background + liver + kidney
    Task 1: Add spleen + pancreas
    Task 2: Add remaining organs
    """
    config = {
        "prefix": "Abdomen Example",
        "dataset": "abdomen",
        "data_dir": "/datasets/abdomen",
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": False,
        "shuffle": False,
        "init_cls": 3,  # Start with background + liver + kidney
        "increment": 2,  # Add two organs at a time
        "model_name": "mote_seg",
        "backbone_type": "swin_unetr_mote_base",
        "device": [0],
        "seed": [1993],

        # Training parameters
        "init_epochs": 150,
        "init_lr": 0.0001,
        "later_epochs": 100,
        "later_lr": 0.00005,
        "batch_size": 2,
        "weight_decay": 1e-5,
        "min_lr": 1e-6,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "pretrained": True,

        # Adapter parameters
        "ffn_num": 64,
        "d_model": 768,
        "alpha": 0.005,
        "use_init_ptm": False,
        "beta": 0,

        # 3D segmentation parameters
        "roi_size": [96, 96, 96],
        "sw_batch_size": 4,
        "overlap": 0.5,
        "use_dice_ce": True,
        "val_interval": 5,

        # Other parameters
        "moni_adam": False,
        "adapter_num": -1,
        "print_forget": True,

        # Dataset-specific
        "num_classes": 14,
        "in_channels": 1,  # CT scan
        "task_type": "segmentation",

        # Computed
        "nb_classes": 14,
        "nb_tasks": 6,
    }

    print("=" * 80)
    print("Example: Abdominal Organ Segmentation with Incremental Learning")
    print("=" * 80)
    print(f"Initial classes: {config['init_cls']}")
    print(f"Increment: {config['increment']}")
    print(f"Total classes: {config['num_classes']}")
    print(f"Backbone: {config['backbone_type']}")
    print(f"ROI size: {config['roi_size']}")
    print("=" * 80)

    # Run training
    train(config)


def example_custom_dataset():
    """
    Example: Custom 3D medical segmentation dataset

    This shows how to configure for a custom dataset
    """
    config = {
        "prefix": "Custom Example",
        "dataset": "medical3d",
        "data_dir": "/path/to/your/dataset",  # CHANGE THIS
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": False,
        "shuffle": False,
        "init_cls": 2,
        "increment": 1,
        "model_name": "mote_seg",
        "backbone_type": "swin_unetr_mote_base",
        "device": [0],
        "seed": [1993],

        # Training parameters
        "init_epochs": 100,
        "init_lr": 0.0001,
        "later_epochs": 50,
        "later_lr": 0.00005,
        "batch_size": 2,
        "weight_decay": 1e-5,
        "min_lr": 1e-6,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "pretrained": True,

        # Adapter parameters
        "ffn_num": 64,
        "d_model": 768,
        "alpha": 0.005,
        "use_init_ptm": False,
        "beta": 0,

        # 3D segmentation parameters
        "roi_size": [96, 96, 96],
        "sw_batch_size": 4,
        "overlap": 0.5,
        "use_dice_ce": True,
        "val_interval": 5,

        # Other parameters
        "moni_adam": False,
        "adapter_num": -1,
        "print_forget": True,

        # Dataset-specific - CUSTOMIZE THESE
        "num_classes": 5,  # Total number of classes including background
        "in_channels": 1,  # Number of input channels (1 for CT, 4 for MRI, etc.)
        "task_type": "segmentation",

        # Computed
        "nb_classes": 5,
        "nb_tasks": 4,
    }

    print("=" * 80)
    print("Example: Custom Medical 3D Segmentation with Incremental Learning")
    print("=" * 80)
    print("IMPORTANT: Update 'data_dir' to point to your dataset!")
    print(f"Initial classes: {config['init_cls']}")
    print(f"Increment: {config['increment']}")
    print(f"Total classes: {config['num_classes']}")
    print(f"Backbone: {config['backbone_type']}")
    print(f"ROI size: {config['roi_size']}")
    print("=" * 80)

    # Uncomment to run:
    # train(config)
    print("\nTo run this example, update the data_dir and uncomment the train() call")


def main():
    """
    Main function to run examples
    """
    print("\nAvailable examples:")
    print("1. BraTS brain tumor segmentation")
    print("2. Abdominal organ segmentation")
    print("3. Custom dataset template")

    # Uncomment the example you want to run:

    # Example 1: BraTS
    # example_brats_segmentation()

    # Example 2: Abdomen
    # example_abdomen_segmentation()

    # Example 3: Custom
    example_custom_dataset()


if __name__ == "__main__":
    main()
