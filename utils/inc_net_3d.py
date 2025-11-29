"""
Incremental Network for 3D Medical Image Segmentation

This module provides network wrappers for incremental learning in
3D medical image segmentation tasks.
"""
import copy
import torch
from torch import nn
from typing import Dict, Optional
from monai.networks.layers import Conv


def get_backbone_3d(args, pretrained=False):
    """
    Get 3D backbone network for medical image segmentation

    Args:
        args: Configuration dictionary
        pretrained: Whether to load pretrained weights

    Returns:
        Backbone model
    """
    name = args["backbone_type"].lower()

    if 'swin_unetr' in name:
        from backbone import swin_unetr_mote
        from easydict import EasyDict

        tuning_config = EasyDict(
            # Adapter configuration
            ffn_adapt=True,
            ffn_option="sequential",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=args.get("ffn_num", 64),
            d_model=args.get("d_model", 768),
            init=args["init_cls"],
            inc=args["increment"],
            _device=args["device"][0]
        )

        if name == "swin_unetr_mote_base":
            model = swin_unetr_mote.swin_unetr_mote_base(
                pretrained=pretrained,
                num_classes=args.get("init_cls", 2),
                tuning_config=tuning_config
            )
            model.out_dim = 768
        elif name == "swin_unetr_mote_small":
            model = swin_unetr_mote.swin_unetr_mote_small(
                pretrained=pretrained,
                num_classes=args.get("init_cls", 2),
                tuning_config=tuning_config
            )
            model.out_dim = 384
        else:
            raise NotImplementedError(f"Unknown backbone type: {name}")

        return model
    else:
        raise NotImplementedError(f"Unknown backbone type: {name}")


class SegmentationHead3D(nn.Module):
    """
    3D Segmentation head for incremental learning

    This head handles incremental class addition for segmentation tasks
    with upsampling to match input resolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        spatial_dims: int = 3,
        upsample_factor: int = 16,  # Default upsampling for Swin UNETR bottleneck
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            kernel_size: Convolution kernel size
            spatial_dims: Spatial dimensions (3 for 3D)
            upsample_factor: Factor to upsample features
        """
        super().__init__()

        self.out_channels = out_channels
        self.upsample_factor = upsample_factor

        # Decoder with progressive upsampling
        # Reduce channels while upsampling
        mid_channels = in_channels // 2

        self.decoder = nn.Sequential(
            # First upsample block
            Conv[Conv.CONV, spatial_dims](in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            # Second upsample block
            Conv[Conv.CONV, spatial_dims](mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            # Third upsample block
            Conv[Conv.CONV, spatial_dims](mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels // 4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            # Fourth upsample block
            Conv[Conv.CONV, spatial_dims](mid_channels // 4, mid_channels // 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels // 8),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
        )

        # Final 1x1 conv for classification
        self.out_conv = Conv[Conv.CONV, spatial_dims](
            in_channels=mid_channels // 8,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W, D) from encoder

        Returns:
            Segmentation logits (B, num_classes, H, W, D) at full resolution
        """
        x = self.decoder(x)
        x = self.out_conv(x)
        return x


class SegMoteNet(nn.Module):
    """
    Network wrapper for incremental 3D medical image segmentation

    This class manages the backbone and segmentation heads for
    continual learning on segmentation tasks.
    """

    def __init__(self, args: Dict, pretrained: bool = False):
        """
        Args:
            args: Configuration dictionary
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        print("Initializing SegMoteNet for 3D segmentation")

        self.args = args
        self.backbone = get_backbone_3d(args, pretrained)
        self._device = args["device"][0]

        # Incremental learning parameters
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1

        # Segmentation head
        self.seg_head = None
        self.proxy_seg_head = None

        # Feature dimension
        self.out_dim = self.backbone.out_dim

        # For testing
        self.use_init_ptm = args.get("use_init_ptm", False)

    @property
    def feature_dim(self):
        """Get feature dimension"""
        return self.out_dim

    def update_seg_head(self, nb_classes: int):
        """
        Update segmentation head for new number of classes

        Args:
            nb_classes: Total number of classes (including new ones)
        """
        self._cur_task += 1

        # Create proxy head for current task
        if self._cur_task == 0:
            # First task
            self.proxy_seg_head = self.generate_seg_head(
                self.out_dim, self.init_cls
            ).to(self._device)
        else:
            # Incremental tasks
            self.proxy_seg_head = self.generate_seg_head(
                self.out_dim, self.inc
            ).to(self._device)

        # Create or expand main segmentation head
        seg_head = self.generate_seg_head(self.out_dim, nb_classes).to(self._device)

        if self.seg_head is not None:
            # Copy weights from previous head
            old_nb_classes = self.seg_head.out_channels
            with torch.no_grad():
                # Copy old class weights
                seg_head.conv.weight.data[:old_nb_classes] = self.seg_head.conv.weight.data
                if seg_head.conv.bias is not None:
                    seg_head.conv.bias.data[:old_nb_classes] = self.seg_head.conv.bias.data

        del self.seg_head
        self.seg_head = seg_head

    def generate_seg_head(self, in_channels: int, out_channels: int) -> SegmentationHead3D:
        """
        Generate a new segmentation head

        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes

        Returns:
            Segmentation head module
        """
        return SegmentationHead3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            spatial_dims=3,
        )

    def forward(self, x: torch.Tensor, test: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W, D)
            test: Whether in test mode

        Returns:
            Dictionary containing:
                - 'logits': Segmentation logits
                - 'features': Intermediate features (optional)
        """
        if not test:
            # Training mode: use current adapter and proxy head
            features = self.backbone.forward_train(x)

            # Apply proxy segmentation head
            logits = self.proxy_seg_head(features)

            return {
                "logits": logits,
                "features": features,
            }
        else:
            # Test mode: may use multiple experts
            features = self.backbone.forward_test(x, use_init_ptm=self.use_init_ptm)

            # Apply full segmentation head
            logits = self.seg_head(features)

            return {
                "logits": logits,
                "features": features,
            }

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        self.backbone.freeze_backbone()

    def add_adapter(self):
        """Add current adapter to list and create new one"""
        self.backbone.add_adapter_to_list()

    def show_trainable_params(self):
        """Print trainable parameters"""
        total_params = 0
        trainable_params = 0

        print("\n=== Trainable Parameters ===")
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"{name}: {param.numel():,}")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    def copy(self):
        """Create a deep copy of the model"""
        return copy.deepcopy(self)

    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
