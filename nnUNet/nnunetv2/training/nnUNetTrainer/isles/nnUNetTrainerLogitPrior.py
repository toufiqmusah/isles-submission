"""
LogitPrior — fuse a spatial prior map into the network output in logit space.

The last input channel is treated as a probability-like map p. Its log-odds transform

    logit(p) = log(p / (1 - p))

is added to the network's logits, scaled by a learnable alpha:

    output = base_logits + alpha * logit(p)

The base network only sees the image channels (the prior channel is split off before
the forward pass). alpha starts at 1 and is learned jointly with the network.

Requirements for this to make sense:
  - the prior channel must be stored un-normalized (NoNormalization in the plans, i.e.
    Dataset103's 'nonorm' channel) so it stays in probability-ish space. Z-scoring it
    (as happened in Dataset102) breaks the log-odds transform.
  - intensity/effect augmentations must not touch the prior channel while geometric
    transforms still act on every channel jointly — see
    nnUNetTrainerPriorProtect.get_training_transforms_protecting_channels.

Only trainers with deep supervision disabled produce the single-tensor output this
wrapper expects (AbstractPrimus sets enable_deep_supervision=False).
"""

import torch
from torch import nn
from typing import List, Union, Tuple

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import (
    nnUNet_PrimusV3S_Trainer_100ep,
    nnUNet_PrimusV3S_Trainer_1250ep,
)
from nnunetv2.training.nnUNetTrainer.isles.nnUNetTrainerPriorProtect import (
    get_training_transforms_protecting_channels,
)


class LogitPriorFusion(nn.Module):
    """
    Wraps a base segmentation network (built for num_input_channels - 1 channels) and
    injects the prior channel as a learnably-scaled log-odds bias on the logits.
    """

    def __init__(self, base_network: nn.Module, prior_eps: float = 1e-4):
        super().__init__()
        self.base = base_network
        self.prior_eps = prior_eps
        self.logit_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if x.shape[1] == 1:
            return self.base(x)
        img = x[:, :-1]
        prior = x[:, -1:]
        prior = torch.clamp(prior, self.prior_eps, 1.0 - self.prior_eps)
        prior_logits = torch.log(prior / (1.0 - prior))
        logits = self.base(img)
        return logits + self.logit_alpha * prior_logits


class nnUNetTrainerLogitPrior(nnUNetTrainer):
    """Default (plans-driven) architecture with a logit-prior fusion head.

    Deep supervision is disabled so the network emits a single tensor, which the
    LogitPriorFusion wrapper requires.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        base = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels - 1,
            num_output_channels,
            enable_deep_supervision=False,
        )
        return LogitPriorFusion(base)

    @staticmethod
    def get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=None, is_cascaded=False, foreground_labels=None, regions=None, ignore_label=None,
    ) -> BasicTransform:
        return get_training_transforms_protecting_channels(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label,
            apply_effects_to_channels=[0],
        )


class nnUNet_PrimusV3S_Trainer_LogitPrior_100ep(nnUNet_PrimusV3S_Trainer_100ep):
    """PrimusV3S 100-epoch trainer with logit-prior fusion (prior channel protected from effect augments)."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        base = nnUNet_PrimusV3S_Trainer_100ep.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels - 1,
            num_output_channels,
            enable_deep_supervision,
        )
        return LogitPriorFusion(base)

    @staticmethod
    def get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=None, is_cascaded=False, foreground_labels=None, regions=None, ignore_label=None,
    ) -> BasicTransform:
        return get_training_transforms_protecting_channels(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label,
            apply_effects_to_channels=[0],
        )


class nnUNet_PrimusV3S_Trainer_LogitPrior_1250ep(nnUNet_PrimusV3S_Trainer_1250ep):
    """PrimusV3S 1250-epoch trainer with logit-prior fusion (prior channel protected from effect augments)."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        base = nnUNet_PrimusV3S_Trainer_1250ep.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels - 1,
            num_output_channels,
            enable_deep_supervision,
        )
        return LogitPriorFusion(base)

    @staticmethod
    def get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=None, is_cascaded=False, foreground_labels=None, regions=None, ignore_label=None,
    ) -> BasicTransform:
        return get_training_transforms_protecting_channels(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label,
            apply_effects_to_channels=[0],
        )
