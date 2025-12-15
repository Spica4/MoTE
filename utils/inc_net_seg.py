"""
Incremental network for 3D medical image segmentation.

Integrates Swin UNETR with MoTE adapters.
"""

import copy
import logging
import torch
from torch import nn


def get_segmentation_backbone(args):
    """Get Swin UNETR backbone with MoTE adapters"""
    from backbone.swin_unetr_mote import swin_unetr_mote
    from easydict import EasyDict

    adapter_config = EasyDict(
        ffn_num=args.get("ffn_num", 64),
        ffn_adapter_init_option=args.get("ffn_adapter_init_option", "lora"),
        ffn_adapter_scalar=args.get("ffn_adapter_scalar", "0.1"),
        ffn_adapter_layernorm_option=args.get("ffn_adapter_layernorm_option", "none"),
    )

    model = swin_unetr_mote(
        img_size=args.get("img_size", (128, 128, 128)),
        in_channels=args.get("in_channels", 1),
        out_channels=args.get("out_channels", 7),  # Will be updated during training
        feature_size=args.get("feature_size", 48),
        adapter_config=adapter_config,
        device=args["device"][0],
    )

    logging.info("Created Swin UNETR with MoTE adapters for medical image segmentation")
    return model


class BaseSegNet(nn.Module):
    """Base network for segmentation"""

    def __init__(self, args, pretrained):
        super(BaseSegNet, self).__init__()

        logging.info('Initializing BaseSegNet for 3D medical image segmentation')
        self.backbone = get_segmentation_backbone(args)
        logging.info('Backbone initialized')

        self._device = args["device"][0]
        self.out_channels = args.get("out_channels", 7)

    @property
    def feature_dim(self):
        return None  # Not applicable for segmentation

    def extract_vector(self, x):
        """Not used for segmentation"""
        return self.backbone(x)

    def forward(self, x, test=False):
        """
        Forward pass through segmentation network.

        Args:
            x: Input tensor [B, C, H, W, D]
            test: Whether in test mode (enables expert selection)

        Returns:
            Segmentation logits [B, num_classes, H, W, D]
        """
        return self.backbone(x, test=test)

    def update_fc(self, nb_classes):
        """Update output channels for new classes"""
        pass  # Output channels are fixed for segmentation

    def freeze(self):
        """Freeze all parameters except adapters"""
        self.backbone.freeze()


class MoteSegNet(BaseSegNet):
    """
    MoTE network for medical image segmentation.

    Integrates Swin UNETR with task-specific adapters for incremental learning.
    """

    def __init__(self, args, pretrained):
        super(MoteSegNet, self).__init__(args, pretrained)

        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.use_init_ptm = args.get("use_init_ptm", False)

    def update_fc(self, nb_classes):
        """
        Update output channels for new number of classes.

        For segmentation, we need to update the final output layer.
        """
        self._cur_task += 1
        self.out_channels = nb_classes

        # Update the output layer of Swin UNETR
        # The base_model.out layer needs to be updated
        old_out = self.backbone.base_model.out

        # Create new output layer with updated number of classes
        from monai.networks.blocks import UnetOutBlock

        new_out = UnetOutBlock(
            spatial_dims=3,
            in_channels=old_out.conv.conv.in_channels,
            out_channels=nb_classes,
        )

        # Copy weights for existing classes if possible
        if self._cur_task > 0:
            old_nb_classes = old_out.conv.conv.out_channels
            with torch.no_grad():
                # Copy weights for old classes
                new_out.conv.conv.weight[:old_nb_classes] = old_out.conv.conv.weight
                if new_out.conv.conv.bias is not None:
                    new_out.conv.conv.bias[:old_nb_classes] = old_out.conv.conv.bias

        # Replace output layer
        self.backbone.base_model.out = new_out.to(self._device)

        logging.info(f"Updated output layer to {nb_classes} classes")

    def forward(self, x, test=False):
        """
        Forward pass through MoTE segmentation network.

        Args:
            x: Input tensor [B, C, H, W, D]
            test: Whether in test mode (enables expert selection)

        Returns:
            Segmentation logits [B, num_classes, H, W, D]
        """
        return self.backbone(x, test=test)

    def freeze(self):
        """Freeze all parameters except current task's adapters"""
        self.backbone.freeze()

    def show_trainable_params(self):
        """Print trainable parameters"""
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                logging.info(f"{name}: {param.numel()}")
                total_params += param.numel()
        logging.info(f"Total trainable parameters: {total_params}")
