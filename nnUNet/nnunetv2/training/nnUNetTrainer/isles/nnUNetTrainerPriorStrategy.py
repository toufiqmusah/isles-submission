from typing import List, Union, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from dynamic_network_architectures.architectures.primus import PrimusV3S
from threadpoolctl import threadpool_limits

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.nnUNetTrainer.isles.nnUNetTrainerPriorProtect import (
    get_training_transforms_protecting_channels,
)
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import (
    nnUNet_PrimusV3S_Trainer_100ep,
    nnUNet_PrimusV3S_Trainer_1250ep,
)


##############################################################################
# 1) ARCHITECTURE: PrimusV3S + spatial attention gate driven by the prior   #
##############################################################################
class PrimusV3SPriorAttentionGate(PrimusV3S):
    """
    A PrimusV3S whose transformer token feature map is conditioned on a spatial
    prior (a separate input channel) via a learnable multiplicative attention gate.

    Prior is injected at the token feature-grid resolution (input / 8) right before
    the decoder (up_projection). The gate is identity-initialized (gate_scale = 0 and
    zero-initialized gate conv), so training starts from the exact baseline Primus
    behaviour and the optimizer learns how strongly to trust the prior.

    x layout (num_input_channels): channel 0 = image, channel 1 = prior.
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate: float = 0.2,
        scale_attn_inner: bool = True,
        init_values: float = 0.1,
    ):
        super().__init__(
            input_channels - 1,   # base net sees only the image channels
            output_channels,
            patch_embed_size,
            input_shape,
            drop_path_rate=drop_path_rate,
            scale_attn_inner=scale_attn_inner,
            init_values=init_values,
        )
        embed_dim = self.embed_dim
        # 3x3 conv over (feature map + downsampled prior) -> feature map shaped delta
        self.gate_conv = nn.Conv3d(embed_dim + 1, embed_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)
        # learnable scalar scaling the (bounded) attention perturbation
        self.gate_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x, ret_mask=False):
        prior = x[:, -1:]          # (B, 1, F, F, F)
        img = x[:, :-1]            # (B, C-1, F, F, F)
        FW, FH, FD = img.shape[2:]

        x = self.down_projection(img)
        B, C, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self.register_tokens is not None:
            x = torch.cat((self.register_tokens.expand(x.shape[0], -1, -1), x), dim=1)
        x, keep_indices = self.eva(x)
        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1]:]

        restored_x, restoration_mask = self.restore_full_sequence(x, keep_indices, num_patches)
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)

        # ---- spatial attention gate from the prior ----
        prior_down = F.adaptive_avg_pool3d(prior, (W, H, D))           # (B, 1, W, H, D)
        delta = self.gate_conv(torch.cat([x, prior_down], dim=1))      # (B, E, W, H, D)
        attention = 1 + self.gate_scale * torch.tanh(delta)           # identity when gate_scale=0
        x = x * attention
        # ------------------------------------------------

        if restoration_mask is not None:
            mask = rearrange(restoration_mask, "b (w h d) -> b w h d", h=H, w=W, d=D)
            full_mask = (
                mask.repeat_interleave(FW // W, dim=1)
                    .repeat_interleave(FH // H, dim=2)
                    .repeat_interleave(FD // D, dim=3)
            )
            full_mask = full_mask[:, None, ...]
        else:
            full_mask = None

        dec_out = self.up_projection(x)
        if ret_mask:
            return dec_out, full_mask
        return dec_out


##############################################################################
# 2) DATALOADER: prior-guided (oversampling) patch selection                 #
##############################################################################
class nnUNetPriorDataLoader(nnUNetDataLoader):
    """
    Extends nnUNetDataLoader so that, on a fraction of the non-foreground-guaranteed
    patches, the crop center is sampled from voxels with high prior value instead of
    uniformly at random. This focuses training patches on the region the prior cares
    about (e.g. the suspected lesion core / penumbra), alongside classic FG oversampling.

    Parameters
    ----------
    prior_channel : channel index holding the spatial prior (default 1).
    prior_subsample_fraction : fraction of *non-force-fg* patches that should be
        recentered on high-prior voxels.
    prior_voxel_fraction : top-`fraction` of voxels by prior value that are candidate
        patch centers.
    """
    def __init__(
        self,
        data,
        batch_size: int,
        patch_size,
        final_patch_size,
        label_manager,
        oversample_foreground_percent: float = 0.33,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling: bool = False,
        transforms=None,
        prior_channel: int = 1,
        prior_subsample_fraction: float = 0.0,
        prior_voxel_fraction: float = 0.02,
    ):
        super().__init__(
            data, batch_size, patch_size, final_patch_size, label_manager,
            oversample_foreground_percent=oversample_foreground_percent,
            sampling_probabilities=sampling_probabilities, pad_sides=pad_sides,
            probabilistic_oversampling=probabilistic_oversampling, transforms=transforms,
        )
        self.prior_channel = prior_channel
        self.prior_subsample_fraction = prior_subsample_fraction
        self.prior_voxel_fraction = prior_voxel_fraction
        self.prior_locations_cache = {}

    def _get_prior_locations(self, identifier, data):
        if identifier in self.prior_locations_cache:
            return self.prior_locations_cache[identifier]
        prior = data[self.prior_channel]
        thr = np.percentile(prior, 100 * (1 - self.prior_voxel_fraction))
        locs = np.argwhere(prior >= thr)
        # leading dummy index mirrors class_locations layout (dummy, z, y, x)
        voxels = [(0, int(z), int(y), int(x)) for z, y, x in locs.tolist()] if len(locs) else []
        self.prior_locations_cache[identifier] = voxels
        return voxels

    def get_bbox(self, data_shape, force_fg, class_locations, prior_locations=None,
                 overwrite_class=None, verbose=False):
        if prior_locations is not None and len(prior_locations) > 0:
            need_to_pad = self.need_to_pad.copy()
            dim = len(data_shape)
            for d in range(dim):
                if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - data_shape[d]
            lbs = [-need_to_pad[i] // 2 for i in range(dim)]
            selected_voxel = prior_locations[np.random.choice(len(prior_locations))]
            bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
            return bbox_lbs, bbox_ubs
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]

                    prior_locations = None
                    if (not force_fg) and (np.random.uniform() < self.prior_subsample_fraction):
                        prior_locations = self._get_prior_locations(i, data)

                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape, force_fg, properties['class_locations'], prior_locations=prior_locations,
                    )
                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)
                    ).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)
                        ).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


##############################################################################
# 3) Shared helpers                                                            #
##############################################################################
class PriorStrategyTrainerMixin:
    """Shared dataloader + transforms behaviour for prior-based trainers."""
    prior_channel = 1

    # dataloader knobs (used only by the oversampling strategy)
    prior_subsample_fraction = 0.0
    prior_voxel_fraction = 0.02

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

    def get_dataloaders(self):
        # build the standard creator path but swap in our dataloader class
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes) = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetPriorDataLoader(
            dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            prior_channel=self.prior_channel,
            prior_subsample_fraction=self.prior_subsample_fraction,
            prior_voxel_fraction=self.prior_voxel_fraction,
        )
        dl_val = nnUNetPriorDataLoader(
            dataset_val, self.batch_size, self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=False,
            prior_channel=self.prior_channel,
            prior_subsample_fraction=0.0,
            prior_voxel_fraction=self.prior_voxel_fraction,
        )
        return dl_tr, dl_val


##############################################################################
# 4) Trainers                                                                   #
##############################################################################
class nnUNet_PrimusV3S_Trainer_PriorAttention_100ep(nnUNet_PrimusV3S_Trainer_100ep):
    """PrimusV3S + spatial prior attention gate, intensity augs off the prior (ch 1)."""
    @staticmethod
    def build_network_architecture(
        plans_manager, configuration_manager, num_input_channels, num_output_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = PrimusV3SPriorAttentionGate(
            num_input_channels, num_output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None) -> BasicTransform:
        return get_training_transforms_protecting_channels(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label,
            apply_effects_to_channels=[0],
        )


class nnUNet_PrimusV3S_Trainer_PriorAttention_1250ep(nnUNet_PrimusV3S_Trainer_1250ep):
    @staticmethod
    def build_network_architecture(
        plans_manager, configuration_manager, num_input_channels, num_output_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = PrimusV3SPriorAttentionGate(
            num_input_channels, num_output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None) -> BasicTransform:
        return get_training_transforms_protecting_channels(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label,
            apply_effects_to_channels=[0],
        )


class nnUNet_PrimusV3S_Trainer_PriorOversample_100ep(PriorStrategyTrainerMixin, nnUNet_PrimusV3S_Trainer_100ep):
    """PrimusV3S (unchanged architecture) trained with prior-guided patch oversampling."""
    prior_subsample_fraction = 0.5


class nnUNet_PrimusV3S_Trainer_PriorOversample_1250ep(PriorStrategyTrainerMixin, nnUNet_PrimusV3S_Trainer_1250ep):
    prior_subsample_fraction = 0.5