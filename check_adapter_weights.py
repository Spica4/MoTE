"""
Check adapter weights to understand if they are properly trained
"""

import torch
import json
import numpy as np

# Load checkpoint
checkpoint_path = "checkpoints/mote_seg/twostage/6/6/task_1_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

state_dict = checkpoint["model_state_dict"]

print("=" * 80)
print("Adapter Weight Analysis")
print("=" * 80)

# Find adapter keys
adapter_keys = [k for k in state_dict.keys() if 'cur_adapter' in k]

if not adapter_keys:
    print("\n❌ No adapter weights found in checkpoint!")
    print("This is a serious problem - adapters should be present.")
    exit(1)

print(f"\nFound {len(adapter_keys)} adapter parameters")

# Debug: Show sample keys to understand structure
print("\nSample adapter keys:")
for key in adapter_keys[:3]:
    print(f"  {key}")

# Group by adapter index
adapters_info = {}
for key in adapter_keys:
    # Extract adapter index from key
    # Expected format: backbone.cur_adapter.0.down_proj.weight
    # or: cur_adapter.0.down_proj.weight
    parts = key.split('.')

    # Find the index after 'cur_adapter'
    try:
        cur_adapter_idx = parts.index('cur_adapter')
        if cur_adapter_idx + 1 < len(parts):
            adapter_idx = int(parts[cur_adapter_idx + 1])
            if adapter_idx not in adapters_info:
                adapters_info[adapter_idx] = []
            adapters_info[adapter_idx].append(key)
    except (ValueError, IndexError):
        continue

print(f"Number of adapters: {len(adapters_info)}")

# Analyze each adapter
print("\n" + "=" * 80)
print("Per-Adapter Analysis:")
print("=" * 80)

for adapter_idx in sorted(adapters_info.keys()):
    keys = adapters_info[adapter_idx]

    # Find down_proj and up_proj weights
    down_proj_weight = None
    down_proj_bias = None
    up_proj_weight = None
    up_proj_bias = None

    for key in keys:
        if 'down_proj.weight' in key:
            down_proj_weight = state_dict[key]
        elif 'down_proj.bias' in key:
            down_proj_bias = state_dict[key]
        elif 'up_proj.weight' in key:
            up_proj_weight = state_dict[key]
        elif 'up_proj.bias' in key:
            up_proj_bias = state_dict[key]

    print(f"\nAdapter {adapter_idx}:")

    if down_proj_weight is not None:
        dw_mean = down_proj_weight.mean().item()
        dw_std = down_proj_weight.std().item()
        dw_abs_mean = down_proj_weight.abs().mean().item()
        dw_max = down_proj_weight.abs().max().item()

        print(f"  down_proj weight shape: {down_proj_weight.shape}")
        print(f"    mean:     {dw_mean:8.5f}")
        print(f"    std:      {dw_std:8.5f}")
        print(f"    |mean|:   {dw_abs_mean:8.5f}")
        print(f"    |max|:    {dw_max:8.5f}")

    if up_proj_weight is not None:
        uw_mean = up_proj_weight.mean().item()
        uw_std = up_proj_weight.std().item()
        uw_abs_mean = up_proj_weight.abs().mean().item()
        uw_max = up_proj_weight.abs().max().item()

        print(f"  up_proj weight shape: {up_proj_weight.shape}")
        print(f"    mean:     {uw_mean:8.5f}")
        print(f"    std:      {uw_std:8.5f}")
        print(f"    |mean|:   {uw_abs_mean:8.5f}")
        print(f"    |max|:    {uw_max:8.5f}")

        # Check if up_proj is still close to zero (LoRA initialization)
        if uw_abs_mean < 0.01:
            print(f"    ⚠️  WARNING: up_proj weights very small (close to initialization)")

# Overall statistics
print("\n" + "=" * 80)
print("Overall Adapter Statistics:")
print("=" * 80)

all_adapter_weights = []
for key in adapter_keys:
    if 'weight' in key:
        all_adapter_weights.append(state_dict[key].abs().mean().item())

if all_adapter_weights:
    avg_weight = np.mean(all_adapter_weights)
    std_weight = np.std(all_adapter_weights)

    print(f"\nAverage |weight| across all adapters: {avg_weight:.5f}")
    print(f"Std deviation: {std_weight:.5f}")

    if avg_weight < 0.05:
        print("\n❌ PROBLEM: Adapter weights are very small!")
        print("   Adapters have not been properly trained.")
        print("   This explains why model cannot predict Task 1 classes.")
        print("\n   Possible causes:")
        print("   - Learning rate too low (currently 0.0001)")
        print("   - Training stopped too early")
        print("   - Gradient flow issues")
    elif avg_weight < 0.1:
        print("\n⚠️  Adapter weights are somewhat small")
        print("   Model may benefit from:")
        print("   - Higher learning rate")
        print("   - More training epochs")
    else:
        print("\n✓ Adapter weights appear reasonable")

# Compare up_proj weights (should grow from ~0 with LoRA init)
print("\n" + "=" * 80)
print("up_proj Weight Growth (LoRA initialization check):")
print("=" * 80)

up_proj_weights = []
for adapter_idx in sorted(adapters_info.keys()):
    for key in adapters_info[adapter_idx]:
        if 'up_proj.weight' in key:
            weight = state_dict[key]
            abs_mean = weight.abs().mean().item()
            up_proj_weights.append(abs_mean)
            print(f"Adapter {adapter_idx} up_proj |mean|: {abs_mean:.5f}")

if up_proj_weights:
    avg_up_proj = np.mean(up_proj_weights)
    print(f"\nAverage up_proj |weight|: {avg_up_proj:.5f}")

    # LoRA initialization sets up_proj to zero
    # After training, it should grow significantly
    if avg_up_proj < 0.01:
        print("\n❌ CRITICAL: up_proj weights still near zero!")
        print("   Adapters have barely changed from initialization.")
        print("   Model is essentially not using adapters at all.")
    elif avg_up_proj < 0.05:
        print("\n⚠️  up_proj weights are small")
        print("   Adapters may need more training")
    else:
        print("\n✓ up_proj weights have grown from initialization")
