"""
Swin UNETR with MoTE (Mixture of Task-specific Experts) for 3D Medical Image Segmentation
Adapted from MONAI's Swin UNETR implementation with MoTE adapter integration
"""

import math
import torch
import torch.nn as nn
import copy
from typing import Optional, Sequence, Tuple, Type, Union
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.utils import ensure_tuple_rep


class Adapter(nn.Module):
    """Adapter module for task-specific fine-tuning in Swin UNETR blocks"""

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 64,
        dropout: float = 0.1,
        init_option: str = "lora",
        adapter_scalar: Union[str, float] = "0.1",
        adapter_layernorm_option: str = "in"
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option

        # Layer normalization
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # Scalar for adapter output
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # Down and up projections
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        # Initialize weights
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class SwinTransformerBlockWithAdapter(nn.Module):
    """
    Swin Transformer Block with integrated MoTE adapter.
    This wraps around MONAI's SwinTransformerBlock and adds adapter functionality.
    """

    def __init__(
        self,
        original_block,
        dim: int,
        adapter_config: dict
    ):
        super().__init__()
        self.original_block = original_block
        self.dim = dim
        self.adapter_config = adapter_config

        # Note: Adapter will be added dynamically during training
        # We don't create it here to allow for task-specific adapters

    def forward(self, x, adapter=None):
        """
        Args:
            x: Input tensor
            adapter: Optional adapter module to apply after the block
        """
        # Apply original Swin Transformer block
        x = self.original_block(x)

        # Apply adapter if provided (for MoTE training/inference)
        if adapter is not None:
            # Reshape for adapter: [B, H, W, D, C] -> [B, H*W*D, C]
            original_shape = x.shape
            B, H, W, D, C = original_shape
            x_flat = x.view(B, H * W * D, C)

            # Apply adapter
            x_adapted = adapter(x_flat, add_residual=False)

            # Add residual and reshape back
            x = x_flat + x_adapted
            x = x.view(original_shape)

        return x


class SwinUNETRWithMoTE(nn.Module):
    """
    Swin UNETR with MoTE (Mixture of Task-specific Experts) for incremental learning.

    This model integrates task-specific adapters into Swin UNETR's transformer blocks,
    allowing for continual learning without catastrophic forgetting.
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int] = (128, 128, 128),
        in_channels: int = 1,
        out_channels: int = 7,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
        adapter_config: Optional[dict] = None,
        device: torch.device = None,
    ):
        super().__init__()

        # Import here to avoid circular dependencies
        from monai.networks.nets import SwinUNETR

        # Create base Swin UNETR model
        self.base_model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
        )

        # MoTE configuration
        self.adapter_config = adapter_config or {
            'ffn_num': 64,
            'ffn_adapter_init_option': 'lora',
            'ffn_adapter_scalar': '0.1',
            'ffn_adapter_layernorm_option': 'none',
        }

        self._device = device
        self.out_channels = out_channels

        # Task tracking
        self.adapter_list = []  # List of adapters for each task
        self.cur_adapter = nn.ModuleList()  # Current task's adapters

        # Feature dimensions for each layer
        self.feature_dims = [
            feature_size,           # layers1c
            feature_size * 2,       # layers2c
            feature_size * 4,       # layers3c
            feature_size * 8,       # layers4c
        ]

        # Wrap transformer blocks with adapter capability
        self._wrap_transformer_blocks()

        # Initialize first adapter
        self.get_new_adapter()

        # Counters for expert selection statistics
        self.multicount = 0
        self.zerocount = 0

    def _wrap_transformer_blocks(self):
        """Wrap Swin Transformer blocks to support adapters"""
        # Access the Swin Transformer blocks through the swinViT encoder
        swin_vit = self.base_model.swinViT

        # We'll apply adapters to the transformer blocks in each layer
        # Layers: layers1, layers2, layers3, layers4
        self.transformer_blocks = []

        for layer_idx, layer_name in enumerate(['layers1', 'layers2', 'layers3', 'layers4']):
            if hasattr(swin_vit, layer_name):
                layer = getattr(swin_vit, layer_name)
                # Each layer contains blocks
                if hasattr(layer, '__iter__'):
                    for module in layer:
                        if hasattr(module, 'blocks'):
                            # This is a BasicLayer with blocks
                            for block_idx, block in enumerate(module.blocks):
                                self.transformer_blocks.append({
                                    'layer_idx': layer_idx,
                                    'block': block,
                                    'dim': self.feature_dims[layer_idx]
                                })

        print(f"Wrapped {len(self.transformer_blocks)} transformer blocks for MoTE adapters")

    def get_new_adapter(self):
        """Create new adapters for the current task"""
        self.cur_adapter = nn.ModuleList()

        # Create one adapter per transformer block
        for block_info in self.transformer_blocks:
            adapter = Adapter(
                d_model=block_info['dim'],
                bottleneck=self.adapter_config['ffn_num'],
                dropout=0.1,
                init_option=self.adapter_config['ffn_adapter_init_option'],
                adapter_scalar=self.adapter_config['ffn_adapter_scalar'],
                adapter_layernorm_option=self.adapter_config['ffn_adapter_layernorm_option'],
            )
            if self._device is not None:
                adapter = adapter.to(self._device)
            self.cur_adapter.append(adapter)

        self.cur_adapter.requires_grad_(True)

    def add_adapter_to_list(self):
        """Save current adapter and create new one for next task"""
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    def freeze(self):
        """Freeze all parameters except current task's adapters"""
        for param in self.parameters():
            param.requires_grad = False

        # Only train current adapters
        for adapter in self.cur_adapter:
            for param in adapter.parameters():
                param.requires_grad = True

    def forward_train(self, x):
        """Forward pass during training with current adapter"""
        # During training, we only use the current adapter
        # We'll need to intercept the forward pass through transformer blocks

        # For training, we use the base model but inject adapters
        # This requires modifying the base model's forward pass
        # For simplicity, we'll use a hook-based approach

        adapter_idx = 0
        hooks = []

        def create_hook(adapter):
            def hook_fn(module, input, output):
                nonlocal adapter_idx
                # Apply adapter to the output
                if adapter_idx < len(self.cur_adapter):
                    # output is the result of the transformer block
                    adapted = self.cur_adapter[adapter_idx](output, add_residual=False)
                    output = output + adapted
                    adapter_idx += 1
                return output
            return hook_fn

        # Register hooks on transformer blocks
        for block_info in self.transformer_blocks:
            hook = block_info['block'].register_forward_hook(create_hook(self.cur_adapter))
            hooks.append(hook)

        try:
            # Forward through base model
            output = self.base_model(x)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return output

    def forward_test(self, x, use_expert_selection=True):
        """
        Forward pass during testing with expert selection/fusion.

        Args:
            x: Input tensor
            use_expert_selection: Whether to use MoTE expert selection
        """
        if not use_expert_selection or len(self.adapter_list) == 0:
            # No expert selection, use current adapter only
            return self.forward_train(x)

        # Get predictions from each expert (adapter)
        expert_outputs = []

        for expert_idx, expert_adapter in enumerate(self.adapter_list):
            adapter_idx = 0
            hooks = []

            def create_hook(adapter):
                def hook_fn(module, input, output):
                    nonlocal adapter_idx
                    if adapter_idx < len(expert_adapter):
                        adapted = expert_adapter[adapter_idx](output, add_residual=False)
                        output = output + adapted
                        adapter_idx += 1
                    return output
                return hook_fn

            # Register hooks
            for block_info in self.transformer_blocks:
                hook = block_info['block'].register_forward_hook(create_hook(expert_adapter))
                hooks.append(hook)

            try:
                # Forward through base model
                with torch.no_grad():
                    output = self.base_model(x)
                expert_outputs.append(output)
            finally:
                for hook in hooks:
                    hook.remove()

        # Merge expert outputs using confidence-based weighting
        # For segmentation, we use softmax confidence across classes
        if len(expert_outputs) == 1:
            return expert_outputs[0]

        # Compute confidence for each expert based on max softmax probability
        expert_confidences = []
        for output in expert_outputs:
            # output shape: [B, C, H, W, D]
            probs = torch.softmax(output, dim=1)
            max_probs = torch.max(probs, dim=1)[0]  # [B, H, W, D]
            confidence = torch.mean(max_probs, dim=[1, 2, 3])  # [B]
            expert_confidences.append(confidence)

        # Stack confidences: [num_experts, B]
        confidences = torch.stack(expert_confidences, dim=0)

        # Find expert with highest confidence for each sample
        max_conf_idx = torch.argmax(confidences, dim=0)  # [B]

        # Weight experts: max confidence expert gets original weight, others * 0.1
        weights = torch.zeros_like(confidences)  # [num_experts, B]
        for b in range(x.shape[0]):
            for e in range(len(expert_outputs)):
                if e == max_conf_idx[b]:
                    weights[e, b] = confidences[e, b]
                else:
                    weights[e, b] = confidences[e, b] * 0.1

        # Normalize weights with softmax
        weights = torch.softmax(weights, dim=0)  # [num_experts, B]

        # Weighted combination of expert outputs
        final_output = torch.zeros_like(expert_outputs[0])
        for e, output in enumerate(expert_outputs):
            # Expand weights to match output shape: [B] -> [B, 1, 1, 1, 1]
            weight = weights[e].view(-1, 1, 1, 1, 1)
            final_output += output * weight

        return final_output

    def forward(self, x, test=False):
        """
        Main forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W, D]
            test: Whether in test mode (enables expert selection)
        """
        if test:
            return self.forward_test(x, use_expert_selection=True)
        else:
            return self.forward_train(x)


def swin_unetr_mote(
    img_size=(128, 128, 128),
    in_channels=1,
    out_channels=7,
    feature_size=48,
    adapter_config=None,
    device=None,
):
    """
    Create Swin UNETR with MoTE adapters for 3D medical image segmentation.

    Args:
        img_size: Input image size (H, W, D)
        in_channels: Number of input channels
        out_channels: Number of output classes (including background)
        feature_size: Base feature dimension
        adapter_config: Configuration for adapters
        device: Device to place the model on

    Returns:
        SwinUNETRWithMoTE model
    """
    model = SwinUNETRWithMoTE(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        adapter_config=adapter_config,
        device=device,
    )

    # Don't freeze during initialization - allow Task 0 to train the backbone
    # Freezing will be done after each task via model.freeze() in the training loop
    # This ensures:
    #   - Task 0: Full backbone + adapters train (backbone learns from scratch)
    #   - Task 1+: Only adapters train (backbone frozen after Task 0)

    return model
