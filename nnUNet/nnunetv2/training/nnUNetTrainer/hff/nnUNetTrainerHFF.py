"""
nnUNet trainer for the HFFNet (Hierarchical Feature Fusion) architecture.

Key differences from base nnUNetTrainer:
  - build_network_architecture: directly instantiates HFFNet (no plans kwargs)
  - Custom train_step / validation_step to handle HFFNet's multi-output:
      training returns (avg, lf, hf) → loss on each, combined
  - set_deep_supervision_enabled: no-op (HFFNet has no deep supervision toggle)
  - enable_deep_supervision = False
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.hff import HFFNet

import torch
from torch import autocast
from typing import Union, List, Tuple

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnUNetTrainerHFF(nnUNetTrainer):
    """
    nnUNet trainer using the original HFFNet architecture.

    Designed for neonatal brain MRI segmentation from low-field CISO images.
    Uses dual-branch LF/HF processing with frequency cross-attention.
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
        self.num_epochs = 500
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        """Directly instantiate HFFNet — no plans kwargs needed."""
        return HFFNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            patch_size=configuration_manager.patch_size,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        """No-op — HFFNet doesn't have a deep supervision toggle."""
        pass

    def _do_i_compile(self):
        """Disable torch.compile — FFT ops crash under inductor/eager fallback."""
        return False

    def _build_loss(self):
        """Use the standard nnUNet loss (DC+CE), but WITHOUT deep supervision wrapper."""
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                  weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # No DeepSupervisionWrapper — we handle multi-output manually
        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            if isinstance(output, tuple):
                # Training mode: (avg_output, lf_output, hf_output)
                avg_out, lf_out, hf_out = output
                # Multi-task loss: equal weight on all three
                l = (self.loss(avg_out, target) +
                     self.loss(lf_out, target) +
                     self.loss(hf_out, target)) / 3.0
            else:
                # Fallback (shouldn't happen during training)
                l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data

            if isinstance(output, tuple):
                avg_out, lf_out, hf_out = output
                l = (self.loss(avg_out, target) +
                     self.loss(lf_out, target) +
                     self.loss(hf_out, target)) / 3.0
                output = avg_out  # use averaged output for online evaluation
            else:
                l = self.loss(output, target)

        # Online evaluation (fake dice / green line) — same as base trainer
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
