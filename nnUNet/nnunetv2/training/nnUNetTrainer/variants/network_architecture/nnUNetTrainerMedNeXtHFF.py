"""
nnUNet trainer using MedNeXt backbone with HFF-style high-frequency dual-route.

For usage, pass `-tr nnUNetTrainerMedNeXtHFF_M` (or _B, _S, _L) to nnUNetv2_train.
"""

from typing import Union, List, Tuple
import torch
from torch._dynamo import OptimizedModule

from nnunet_mednext import create_mednext_v1
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.mednext_hff import MedNeXtHFF


class nnUNetTrainerMedNeXtHFF(nnUNetTrainer):
    model_id = 'M'
    kernel_size = 3

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold,
                         dataset_json=dataset_json, device=device)
        self.num_epochs = 500

    def _get_base_model(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def _do_i_compile(self):
        return False

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._get_base_model()
        if hasattr(mod, 'mednext'):
            # mednext checks 'do_ds' (not 'deep_supervision') to decide
            # whether forward returns a list (DS) or single tensor.
            mod.mednext.do_ds = enabled
            mod.do_ds = enabled

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                    num_input_channels: int, num_output_channels: int,
                                    enable_deep_supervision: bool = True) -> torch.nn.Module:
        # Always build with deep_supervision=True so that out_1–out_4 heads
        # exist in the architecture and checkpoint weights can be loaded.
        mednext = create_mednext_v1(
            num_input_channels=num_input_channels,
            num_classes=num_output_channels,
            model_id='M',
            kernel_size=3,
            deep_supervision=True,
        )
        wrapper = MedNeXtHFF(mednext)
        # mednext checks 'do_ds' (not 'deep_supervision') to decide
        # whether forward returns a list (DS) or single tensor.
        wrapper.do_ds = enable_deep_supervision
        mednext.do_ds = enable_deep_supervision
        return wrapper


class nnUNetTrainerMedNeXtHFF_S(nnUNetTrainerMedNeXtHFF):
    model_id = 'S'
    kernel_size = 3


class nnUNetTrainerMedNeXtHFF_B(nnUNetTrainerMedNeXtHFF):
    model_id = 'B'
    kernel_size = 3


class nnUNetTrainerMedNeXtHFF_M(nnUNetTrainerMedNeXtHFF):
    model_id = 'M'
    kernel_size = 3


class nnUNetTrainerMedNeXtHFF_L(nnUNetTrainerMedNeXtHFF):
    model_id = 'L'
    kernel_size = 3


class nnUNetTrainerMedNeXtHFF_M_5(nnUNetTrainerMedNeXtHFF):
    model_id = 'M'
    kernel_size = 5


class nnUNetTrainerMedNeXtHFF_B_5(nnUNetTrainerMedNeXtHFF):
    model_id = 'B'
    kernel_size = 5


class nnUNetTrainerMedNeXtHFF_S_5(nnUNetTrainerMedNeXtHFF):
    model_id = 'S'
    kernel_size = 5


class nnUNetTrainerMedNeXtHFF_L_5(nnUNetTrainerMedNeXtHFF):
    model_id = 'L'
    kernel_size = 5
