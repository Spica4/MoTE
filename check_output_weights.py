"""
Check output layer weights to understand why model predicts only background
"""

import torch
import json
import numpy as np

# Load checkpoint
checkpoint_path = "checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

state_dict = checkpoint["model_state_dict"]

# Get output layer weights
out_weight_key = "backbone.base_model.out.conv.conv.weight"
out_bias_key = "backbone.base_model.out.conv.conv.bias"

if out_weight_key in state_dict:
    out_weight = state_dict[out_weight_key]  # [13, 48, 1, 1, 1]
    out_bias = state_dict[out_bias_key]      # [13]

    print("=" * 80)
    print("Output Layer Analysis")
    print("=" * 80)
    print(f"\nOutput weight shape: {out_weight.shape}")
    print(f"Output bias shape: {out_bias.shape}")

    # Analyze weights for each class
    print("\n" + "=" * 80)
    print("Per-Class Weight Statistics:")
    print("=" * 80)

    class_names = [
        "背景", "大動脈", "食道", "肝臓", "胆嚢", "胃", "脾臓",
        "右腎臓", "左腎臓", "下大動脈", "膵臓", "膀胱", "子宮"
    ]

    for cls in range(13):
        weight_cls = out_weight[cls]  # [48, 1, 1, 1]
        bias_cls = out_bias[cls]

        w_mean = weight_cls.mean().item()
        w_std = weight_cls.std().item()
        w_abs_mean = weight_cls.abs().mean().item()
        b_val = bias_cls.item()

        task = "Task 0" if cls <= 6 else "Task 1"
        print(f"\nClass {cls:2d} ({class_names[cls]:8s}) [{task}]:")
        print(f"  Weight mean: {w_mean:8.5f}")
        print(f"  Weight std:  {w_std:8.5f}")
        print(f"  Weight |mean|: {w_abs_mean:8.5f}")
        print(f"  Bias:        {b_val:8.5f}")

    # Compare Task 0 vs Task 1 classes
    print("\n" + "=" * 80)
    print("Task 0 vs Task 1 Comparison:")
    print("=" * 80)

    task0_weights = out_weight[0:7].abs().mean().item()
    task1_weights = out_weight[7:13].abs().mean().item()

    task0_bias = out_bias[0:7].abs().mean().item()
    task1_bias = out_bias[7:13].abs().mean().item()

    print(f"\nTask 0 classes (0-6):")
    print(f"  Avg |weight|: {task0_weights:.5f}")
    print(f"  Avg |bias|:   {task0_bias:.5f}")

    print(f"\nTask 1 classes (7-12):")
    print(f"  Avg |weight|: {task1_weights:.5f}")
    print(f"  Avg |bias|:   {task1_bias:.5f}")

    ratio = task0_weights / task1_weights if task1_weights > 0 else float('inf')
    print(f"\nRatio (Task 0 / Task 1): {ratio:.2f}x")

    if ratio > 10:
        print("\n❌ PROBLEM: Task 1 weights are much smaller than Task 0!")
        print("   This explains why model doesn't predict Task 1 classes.")
        print("   The output layer for Task 1 classes is not properly trained.")
    elif task1_weights < 0.01:
        print("\n⚠️  WARNING: Task 1 weights are very small")
        print("   Model may have difficulty predicting Task 1 classes")
    else:
        print("\n✓ Task 1 weights seem reasonable")

    # Check if background dominates
    print("\n" + "=" * 80)
    print("Background Bias Analysis:")
    print("=" * 80)

    bg_bias = out_bias[0].item()
    fg_bias_mean = out_bias[1:].mean().item()

    print(f"Background (class 0) bias: {bg_bias:.5f}")
    print(f"Foreground classes bias (avg): {fg_bias_mean:.5f}")
    print(f"Difference: {bg_bias - fg_bias_mean:.5f}")

    if bg_bias > fg_bias_mean + 1.0:
        print("\n⚠️  Background bias is much larger than foreground")
        print("   This makes the model prefer background predictions")

else:
    print("Output layer weights not found in checkpoint!")
