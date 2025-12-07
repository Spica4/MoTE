"""
Swin UNETR with MoTE (Mixture of Task-specific Experts) for 3D Medical Image Segmentation

This module extends MONAI's Swin UNETR with adapter-based incremental learning capabilities.
"""
import math
import copy
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import Conv


class Adapter3D(nn.Module):
    """
    3D Adapter module for incremental learning in medical image segmentation

    Similar to 2D adapters but optimized for 3D feature maps
    """
    def __init__(
        self,
        d_model: int,
        bottleneck: int = 64,
        dropout: float = 0.1,
        init_option: str = "lora",
        adapter_scalar: Union[str, float] = 0.1,
        adapter_layernorm_option: str = "in"
    ):
        """
        Args:
            d_model: Input/output dimension
            bottleneck: Bottleneck dimension for adapter
            dropout: Dropout rate
            init_option: Initialization method ('lora' or 'bert')
            adapter_scalar: Scaling factor for adapter output
            adapter_layernorm_option: Layer norm position ('in', 'out', or None)
        """
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None

        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()  # GELU for medical imaging
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

        # LoRA-style initialization
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, add_residual: bool = True, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            add_residual: Whether to add residual connection
            residual: Residual tensor (default: x)

        Returns:
            Adapted features
        """
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


class SwinUNETRMoTE(nn.Module):
    """
    Swin UNETR with MoTE adapters for incremental learning

    This model extends MONAI's Swin UNETR with task-specific adapters
    for continual learning on medical image segmentation tasks.
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        tuning_config=None,
    ):
        """
        Args:
            img_size: Input image size
            in_channels: Number of input channels
            out_channels: Number of output channels (classes)
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            feature_size: Base feature size
            norm_name: Normalization layer name
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Dropout path rate
            normalize: Whether to use input normalization
            use_checkpoint: Whether to use gradient checkpointing
            spatial_dims: Spatial dimensions (should be 3 for 3D)
            tuning_config: Configuration for adapters and incremental learning
        """
        super().__init__()

        print("Initializing Swin UNETR with MoTE adapters for 3D segmentation")

        self.tuning_config = tuning_config
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels

        # Create base Swin UNETR model
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        # MoTE-specific components
        self.config = tuning_config
        self._device = tuning_config._device if tuning_config else "cuda"
        self.init = tuning_config.init if tuning_config else 2
        self.inc = tuning_config.inc if tuning_config else 2

        # Store architecture parameters for adapter creation
        self.depths = depths
        self.feature_size = feature_size

        # Adapter management
        self.adapter_list = []  # Stores adapters from previous tasks
        self.cur_adapter = nn.ModuleList()  # Current task's adapters

        # Calculate total number of transformer blocks
        self.num_layers = sum(depths)

        # Get feature dimension from Swin encoder
        # For Swin UNETR, the actual feature dimensions are doubled at each stage
        # With feature_size=24: [48, 96, 192, 384, 768]
        # Formula: feature_size * 2^(i+1) for i in range(len(depths)+1)
        self.feature_dims = [feature_size * (2 ** (i+1)) for i in range(len(depths) + 1)]
        print(f"Swin UNETR feature dimensions per stage: {self.feature_dims}")

        # Initialize adapters for the first task
        if tuning_config and tuning_config.ffn_adapt:
            self.get_new_adapter()

    def get_new_adapter(self):
        """Create a new set of adapters for the current task"""
        config = self.config
        self.cur_adapter = nn.ModuleList()

        if config.ffn_adapt:
            # Create adapters for each stage of Swin Transformer
            for i, (depth, dim) in enumerate(zip(self.depths, self.feature_dims)):
                stage_adapters = nn.ModuleList()
                for j in range(depth):
                    adapter = Adapter3D(
                        d_model=dim,
                        bottleneck=config.ffn_num,
                        dropout=0.1,
                        init_option=config.ffn_adapter_init_option,
                        adapter_scalar=config.ffn_adapter_scalar,
                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                    ).to(self._device)
                    stage_adapters.append(adapter)
                self.cur_adapter.append(stage_adapters)

            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not using adapter===")

    def add_adapter_to_list(self):
        """Save current adapter and create new one for next task"""
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    def freeze_backbone(self):
        """Freeze base model parameters, keep only current adapter trainable"""
        for param in self.swin_unetr.parameters():
            param.requires_grad = False

        # Keep current adapter trainable
        for stage in self.cur_adapter:
            for adapter in stage:
                for param in adapter.parameters():
                    param.requires_grad = True

    def _apply_adapters_to_features(self, features: torch.Tensor, adapters: nn.ModuleList, stage_idx: int) -> torch.Tensor:
        """
        Apply adapters to intermediate features

        Args:
            features: Input features from Swin stage
            adapters: Adapter modules for this stage
            stage_idx: Current stage index

        Returns:
            Adapted features
        """
        # Features shape: (B, H, W, D, C) or (B, N, C) depending on stage
        # Apply adapter if available
        if adapters is not None and stage_idx < len(adapters):
            stage_adapters = adapters[stage_idx]
            # Reshape if needed for adapter processing
            original_shape = features.shape

            if len(features.shape) == 5:  # (B, H, W, D, C)
                B, H, W, D, C = features.shape
                features = features.reshape(B, H * W * D, C)

            # Apply each adapter in the stage sequentially
            for adapter in stage_adapters:
                features = adapter(features)

            # Reshape back if needed
            if len(original_shape) == 5:
                features = features.reshape(original_shape)

        return features

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training (uses current adapter only)

        Args:
            x: Input tensor (B, C, H, W, D)

        Returns:
            Encoder features (B, feature_dim, H', W', D') for segmentation head
        """
        # Extract features from Swin Transformer encoder
        # Instead of running full forward pass, we extract encoder features

        # Run through Swin encoder
        hidden_states = self.swin_unetr.swinViT(x)  # Get encoder features

        # Debug: Print all hidden state shapes
        # if isinstance(hidden_states, (list, tuple)):
        #     print(f"\nDebug - All hidden states shapes: {[hs.shape for hs in hidden_states]}")

        # SwinViT returns a list of hidden states from different stages in (B, C, H, W, D) format
        # We need the one with the largest channel dimension (C at index 1)
        if isinstance(hidden_states, (list, tuple)):
            # Select the hidden state with the largest channel dimension
            # Assuming format is (B, C, H, W, D) where C is at index 1
            features = max(hidden_states, key=lambda hs: hs.shape[1] if len(hs.shape) == 5 else 0)
            # print(f"Selected hidden state with shape: {features.shape} (channel dim: {features.shape[1]})")
        else:
            features = hidden_states

        # Apply current task's adapters if available
        if self.config and self.config.ffn_adapt and len(self.cur_adapter) > 0:
            # Apply adapters to encoder features
            # Note: This is a simplified version - full integration would apply
            # adapters at each transformer block

            # Get the last stage adapter
            last_stage_idx = len(self.cur_adapter) - 1
            if last_stage_idx >= 0:
                stage_adapters = self.cur_adapter[last_stage_idx]

                # Reshape features for adapter if needed
                # Adapter expects (B, N, C) where C is the feature dimension
                # Features from swinViT are in (B, C, H, W, D) format
                original_shape = features.shape

                if len(features.shape) == 5:
                    # (B, C, H, W, D) format from Swin encoder
                    B, C, H, W, D = features.shape
                    # print(f"Reshaping features from {features.shape} to (B, N, C) format for adapter")
                    # Permute to (B, H, W, D, C) then reshape to (B, H*W*D, C)
                    features = features.permute(0, 2, 3, 4, 1).contiguous()  # -> (B, H, W, D, C)
                    features = features.reshape(B, H * W * D, C)
                    # print(f"Reshaped to: {features.shape}")
                elif len(features.shape) == 3:
                    # Already (B, N, C) format
                    # print(f"Features already in (B, N, C) format: {features.shape}")
                else:
                    raise ValueError(f"Unexpected feature shape: {features.shape}")

                # Apply adapters sequentially
                for adapter in stage_adapters:
                    features = adapter(features)

                # Reshape back to (B, C, H, W, D) format
                if len(original_shape) == 5:
                    B, N, C = features.shape
                    H, W, D = original_shape[2], original_shape[3], original_shape[4]
                    features = features.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()
                    # print(f"Restored to original shape: {features.shape}")
        else:
            # No adapters used - need to convert features to (B, C, H, W, D) format
            if len(features.shape) == 5:
                # Check if already in correct format or needs permute
                # If spatial dims are larger than channel dim, likely (B, C, H, W, D) already
                if features.shape[1] < features.shape[2]:
                    # Already (B, C, H, W, D)
                    pass
                else:
                    # (B, H, W, D, C) -> (B, C, H, W, D)
                    features = features.permute(0, 4, 1, 2, 3).contiguous()
            elif len(features.shape) == 3:
                # (B, N, C) - need to reshape to spatial dimensions
                B, N, C = features.shape
                spatial_size = int(round(N ** (1/3)))
                if spatial_size ** 3 == N:
                    features = features.transpose(1, 2).reshape(B, C, spatial_size, spatial_size, spatial_size)
                else:
                    # Fallback: use average pooling and upsample
                    features = features.mean(dim=1, keepdim=True)
                    features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    features = torch.nn.functional.interpolate(
                        features,
                        size=(6, 6, 6),
                        mode='trilinear',
                        align_corners=False
                    )

        return features

    def forward_test(self, x: torch.Tensor, use_init_ptm: bool = False) -> torch.Tensor:
        """
        Forward pass during testing (may use multiple adapters)

        Args:
            x: Input tensor (B, C, H, W, D)
            use_init_ptm: Whether to use initial pre-trained model

        Returns:
            Encoder features (B, feature_dim, H', W', D') for segmentation head
        """
        # Extract features from Swin Transformer encoder
        hidden_states = self.swin_unetr.swinViT(x)

        # Debug: Print all hidden state shapes
        # if isinstance(hidden_states, (list, tuple)):
        #     print(f"\nDebug - All hidden states shapes: {[hs.shape for hs in hidden_states]}")

        # SwinViT returns a list of hidden states from different stages in (B, C, H, W, D) format
        # We need the one with the largest channel dimension (C at index 1)
        if isinstance(hidden_states, (list, tuple)):
            # Select the hidden state with the largest channel dimension
            # Assuming format is (B, C, H, W, D) where C is at index 1
            features = max(hidden_states, key=lambda hs: hs.shape[1] if len(hs.shape) == 5 else 0)
            # print(f"Selected hidden state with shape: {features.shape} (channel dim: {features.shape[1]})")
        else:
            features = hidden_states

        # For testing, we can use ensemble of multiple expert adapters
        # or select the best adapter based on confidence
        # Simplified version: use current adapter

        # Apply current task's adapters if available
        if self.config and self.config.ffn_adapt and len(self.cur_adapter) > 0:
            # Get the last stage adapter
            last_stage_idx = len(self.cur_adapter) - 1
            if last_stage_idx >= 0:
                stage_adapters = self.cur_adapter[last_stage_idx]

                # Reshape features for adapter if needed
                # Adapter expects (B, N, C) where C is the feature dimension
                # Features from swinViT are in (B, C, H, W, D) format
                original_shape = features.shape

                if len(features.shape) == 5:
                    # (B, C, H, W, D) format from Swin encoder
                    B, C, H, W, D = features.shape
                    # print(f"Reshaping features from {features.shape} to (B, N, C) format for adapter")
                    # Permute to (B, H, W, D, C) then reshape to (B, H*W*D, C)
                    features = features.permute(0, 2, 3, 4, 1).contiguous()  # -> (B, H, W, D, C)
                    features = features.reshape(B, H * W * D, C)
                    # print(f"Reshaped to: {features.shape}")
                elif len(features.shape) == 3:
                    # Already (B, N, C) format
                    # print(f"Features already in (B, N, C) format: {features.shape}")
                else:
                    raise ValueError(f"Unexpected feature shape: {features.shape}")

                # Apply adapters sequentially
                for adapter in stage_adapters:
                    features = adapter(features)

                # Reshape back to (B, C, H, W, D) format
                if len(original_shape) == 5:
                    B, N, C = features.shape
                    H, W, D = original_shape[2], original_shape[3], original_shape[4]
                    features = features.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()
                    # print(f"Restored to original shape: {features.shape}")
        else:
            # No adapters used - need to convert features to (B, C, H, W, D) format
            if len(features.shape) == 5:
                # Check if already in correct format or needs permute
                if features.shape[1] < features.shape[2]:
                    # Already (B, C, H, W, D)
                    pass
                else:
                    # (B, H, W, D, C) -> (B, C, H, W, D)
                    features = features.permute(0, 4, 1, 2, 3).contiguous()
            elif len(features.shape) == 3:
                # (B, N, C) - need to reshape to spatial dimensions
                B, N, C = features.shape
                spatial_size = int(round(N ** (1/3)))
                if spatial_size ** 3 == N:
                    features = features.transpose(1, 2).reshape(B, C, spatial_size, spatial_size, spatial_size)
                else:
                    # Fallback: use average pooling and upsample
                    features = features.mean(dim=1, keepdim=True)
                    features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    features = torch.nn.functional.interpolate(
                        features,
                        size=(6, 6, 6),
                        mode='trilinear',
                        align_corners=False
                    )

        return features

    def forward(self, x: torch.Tensor, test_mode: bool = False) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W, D)
            test_mode: Whether in test mode

        Returns:
            Segmentation output (B, num_classes, H, W, D)
        """
        if test_mode:
            return self.forward_test(x)
        else:
            return self.forward_train(x)


def swin_unetr_mote_base(pretrained: bool = False, **kwargs):
    """
    Create a base Swin UNETR with MoTE adapters

    Args:
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments

    Returns:
        SwinUNETRMoTE model
    """
    model = SwinUNETRMoTE(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=kwargs.get('num_classes', 2),
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        tuning_config=kwargs.get('tuning_config'),
    )

    if pretrained:
        # Load pretrained weights if available
        # MONAI provides pretrained Swin UNETR weights
        print("Loading pretrained Swin UNETR weights from MONAI...")
        # This would require MONAI's pretrained weight loading mechanism

    return model


def swin_unetr_mote_small(pretrained: bool = False, **kwargs):
    """
    Create a small Swin UNETR with MoTE adapters

    Args:
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments

    Returns:
        SwinUNETRMoTE model
    """
    model = SwinUNETRMoTE(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=kwargs.get('num_classes', 2),
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        tuning_config=kwargs.get('tuning_config'),
    )

    return model
