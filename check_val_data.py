"""
Quick script to check validation data labels only (no model needed)
"""

import json
import numpy as np
from utils.medical_data_manager import MedicalDataManager
from torch.utils.data import DataLoader

# Load config
with open("exps/medical/twostage_swin_unetr_task1.json", 'r') as f:
    args = json.load(f)

# Create data manager
data_manager = MedicalDataManager(
    args["dataset"],
    args["shuffle"],
    args["seed"][0],
    args["init_cls"],
    args["increment"],
    args,
)

# Prepare for Task 1
data_manager.prepare_task(1)

# Get validation dataset
val_classes = np.arange(0, 12)
val_dataset = data_manager.get_dataset(val_classes, source="val", mode="test")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

print(f"\n{'='*80}")
print(f"Validation Dataset: {len(val_dataset)} samples")
print(f"{'='*80}\n")

class_counts = np.zeros(13, dtype=np.int64)

# Check first 10 samples
for idx, (_, inputs, targets) in enumerate(val_loader):
    if idx >= 10:
        break

    targets_np = targets.cpu().numpy()
    unique, counts = np.unique(targets_np, return_counts=True)

    print(f"Sample {idx + 1}:")
    print(f"  Unique classes: {unique}")

    for cls, count in zip(unique, counts):
        if cls < 13:
            class_counts[int(cls)] += count
        percentage = count / targets_np.size * 100
        if percentage > 1.0:  # Show only classes with > 1%
            print(f"    Class {int(cls)}: {percentage:.1f}%")

print(f"\n{'='*80}")
print("Overall Distribution:")
print(f"{'='*80}")

total = class_counts.sum()
for cls in range(13):
    if class_counts[cls] > 0:
        pct = class_counts[cls] / total * 100
        names = ["背景", "大動脈", "食道", "肝臓", "胆嚢", "胃", "脾臓",
                 "右腎臓", "左腎臓", "下大動脈", "膵臓", "膀胱", "子宮"]
        print(f"Class {cls:2d} ({names[cls]:8s}): {pct:6.2f}%")

# Check Task 1 classes
task1_pixels = class_counts[7:13].sum()
bg_pixels = class_counts[0]

print(f"\n{'='*80}")
print("Analysis:")
print(f"{'='*80}")
print(f"Background (class 0): {bg_pixels/total*100:.2f}%")
print(f"Task 1 classes (7-12): {task1_pixels/total*100:.2f}%")

if task1_pixels == 0:
    print("\n❌ PROBLEM: No Task 1 classes found!")
    print("   This explains why Val Dice = 0.0000")
elif task1_pixels / total < 0.01:
    print(f"\n⚠️  WARNING: Task 1 classes < 1% of data")
else:
    print(f"\n✓ Task 1 classes present")
