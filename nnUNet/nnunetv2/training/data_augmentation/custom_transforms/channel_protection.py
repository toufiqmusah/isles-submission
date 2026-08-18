from typing import List, Union

import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class ApplyTransformsToChannels(BasicTransform):
    """
    Runs a sequence of transforms but only lets them touch the channels listed in
    `apply_to_channels`. All other channels are temporarily parked (zeroed) while the
    inner transforms run and are restored unchanged afterwards.

    Use case: multi-channel inputs where some channels carry semantics that must not be
    corrupted by intensity augmentations (e.g. a spatial prior / probability map).
    Geometric transforms (rotation, scaling, mirroring) must stay OUTSIDE this wrapper so
    that all channels still move together.
    """
    def __init__(self, transforms: List[BasicTransform], apply_to_channels: Union[List[int], None] = None):
        super().__init__()
        self.transforms = transforms
        self.apply_to_channels = apply_to_channels

    def apply(self, data_dict: dict, **params) -> dict:
        img = data_dict.get('image')
        if img is None or self.apply_to_channels is None or len(self.apply_to_channels) == 0:
            for t in self.transforms:
                data_dict = t(**data_dict)
            return data_dict

        n_channels = img.shape[0]
        apply = [c for c in self.apply_to_channels if 0 <= c < n_channels]
        protected = [c for c in range(n_channels) if c not in apply]
        if not protected or not apply:
            # nothing to protect or nothing to transform; run inner as-is
            for t in self.transforms:
                data_dict = t(**data_dict)
            return data_dict

        with torch.no_grad():
            saved = img[protected].clone()
            img[protected] = 0

        for t in self.transforms:
            data_dict = t(**data_dict)

        with torch.no_grad():
            data_dict['image'][protected] = saved
        return data_dict
