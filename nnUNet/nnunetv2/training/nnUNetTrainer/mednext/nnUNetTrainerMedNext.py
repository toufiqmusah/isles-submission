import os
import torch
from torch._dynamo import OptimizedModule

from nnunet_mednext import create_mednext_v1
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)


def trilinear_kernel_expand(
    state_dict: dict,
    target_kernel_size: int,
    source_kernel_size: int = 3,
) -> dict:
    """
    Expand only those convolutional kernels whose spatial extent matches
    ``source_kernel_size`` (default 3) to ``target_kernel_size`` via
    trilinear/bilinear interpolation.  1×1×1 pointwise convs, biases, and
    norm params are copied unchanged.
    """
    expanded = {}
    ndim = None
    for k, v in state_dict.items():
        if "bias" in k or "norm" in k or "dummy" in k:
            expanded[k] = v
        else:
            shape = v.shape
            if len(shape) >= 3:
                out_c, in_c, *spatial = shape
                if ndim is None:
                    ndim = len(spatial)
                source_spatial = (source_kernel_size,) * ndim
                target_spatial = (target_kernel_size,) * ndim
                if tuple(spatial) == source_spatial:
                    mode = "trilinear" if ndim == 3 else "bilinear"
                    expanded[k] = torch.nn.functional.interpolate(
                        v, size=target_spatial, mode=mode,
                    )
                else:
                    expanded[k] = v
            else:
                expanded[k] = v
    return expanded


def expand_kernel_checkpoint(
    input_checkpoint: str,
    output_checkpoint: str,
    from_k: int = 3,
    to_k: int = 5,
) -> None:
    """
    Load a checkpoint saved by ``nnUNetTrainer.save_checkpoint``, expand
    all convolutional kernels from ``from_k`` → ``to_k`` via trilinear /
    bilinear interpolation, and save the result as a valid
    ``-pretrained_weights`` file for the k5 variant.
    """
    ckpt = torch.load(input_checkpoint, map_location="cpu", weights_only=False)
    weights = ckpt.get("network_weights", ckpt.get("state_dict", ckpt))
    expanded = trilinear_kernel_expand(weights, target_kernel_size=to_k)

    # Build a minimal checkpoint that nnUNet's load_pretrained_weights accepts
    output = {"network_weights": expanded}
    torch.save(output, output_checkpoint)
    print(
        f"Expanded kernels {from_k}→{to_k}: "
        f"{len(expanded)} keys written to {output_checkpoint}"
    )


class nnUNetTrainerMedNext(nnUNetTrainer):
    model_id = "M"
    kernel_size = 3

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000

    def _get_base_model(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._get_base_model()
        if hasattr(mod, "do_ds"):
            mod.do_ds = enabled

    def _do_i_compile(self):
        return False

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        model = create_mednext_v1(
            num_input_channels=num_input_channels,
            num_classes=num_output_channels,
            model_id=cls.model_id,
            kernel_size=cls.kernel_size,
            deep_supervision=True,
        )
        model.do_ds = enable_deep_supervision
        # gradient checkpointing adds 30-40% overhead and is unnecessary for
        # MedNeXt (~17M params); disable regardless of factory default
        if hasattr(model, 'outside_block_checkpointing'):
            model.outside_block_checkpointing = False
        if hasattr(model, 'inside_block_checkpointing'):
            model.inside_block_checkpointing = False
        return model


# ── Model size / kernel variants ────────────────────────────────────────


class nnUNetTrainerMedNext_S(nnUNetTrainerMedNext):
    model_id = "S"
    kernel_size = 3


class nnUNetTrainerMedNext_B(nnUNetTrainerMedNext):
    model_id = "B"
    kernel_size = 3


class nnUNetTrainerMedNext_M(nnUNetTrainerMedNext):
    model_id = "M"
    kernel_size = 3


class nnUNetTrainerMedNext_L(nnUNetTrainerMedNext):
    model_id = "L"
    kernel_size = 3


class nnUNetTrainerMedNext_M_k5(nnUNetTrainerMedNext):
    model_id = "M"
    kernel_size = 5


class nnUNetTrainerMedNext_S_k5(nnUNetTrainerMedNext):
    model_id = "S"
    kernel_size = 5


class nnUNetTrainerMedNext_B_k5(nnUNetTrainerMedNext):
    model_id = "B"
    kernel_size = 5


class nnUNetTrainerMedNext_L_k5(nnUNetTrainerMedNext):
    model_id = "L"
    kernel_size = 5


# ── k5 from k3 via kernel expansion (trilinear upsampling) ──────────────


class nnUNetTrainerMedNext_M_k5_from_k3(nnUNetTrainerMedNext_M_k5):
    """
    Build a k5 MedNeXt-M and load a k3 checkpoint with trilinear kernel
    expansion.  Set the environment variable ``MEDNEXT_K3_CHECKPOINT`` to
    the path of a k3 ``checkpoint_best.pth`` (or ``checkpoint_final.pth``).

    Usage::

        export MEDNEXT_K3_CHECKPOINT=/path/to/k3/fold_X/checkpoint_best.pth
        nnUNetv2_train 102 3d_fullres X -tr nnUNetTrainerMedNext_M_k5_from_k3

    The checkpoint is loaded once during ``initialize()``; no
    ``-pretrained_weights`` flag is needed.
    """

    def _expand_from_k3(self, k3_path: str) -> None:
        mod = self._get_base_model()
        state = torch.load(k3_path, map_location="cpu", weights_only=False)
        src = state.get("network_weights", state.get("state_dict", state))
        expanded = trilinear_kernel_expand(src, target_kernel_size=5)
        # Filter only keys that exist in k5 network (DS heads may differ)
        model_dict = mod.state_dict()
        compat = {k: v for k, v in expanded.items() if k in model_dict}
        missing = [k for k in expanded if k not in model_dict]
        if missing:
            self.print_to_log_file(
                f"Ignored {len(missing)} keys not in k5 network "
                f"(e.g. DS heads)."
            )
        model_dict.update(compat)
        mod.load_state_dict(model_dict)
        n_loaded = len(compat)
        n_expanded = sum(
            1 for k in compat if src[k].shape != model_dict[k].shape
        )
        self.print_to_log_file(
            f"Loaded {n_loaded} keys from k3 checkpoint, "
            f"{n_expanded} kernel-expanded trilinearly."
        )

    def initialize(self):
        if not self.was_initialized:
            super().initialize()
            k3_path = os.environ.get("MEDNEXT_K3_CHECKPOINT")
            if k3_path is not None:
                if not os.path.isfile(k3_path):
                    raise FileNotFoundError(
                        f"MEDNEXT_K3_CHECKPOINT={k3_path} not found"
                    )
                self.print_to_log_file(
                    f"Expanding k3 checkpoint → k5: {k3_path}"
                )
                self._expand_from_k3(k3_path)


# ── Epoch variants (M, k=3) ─────────────────────────────────────────────


class nnUNetTrainerMedNext_50epochs(nnUNetTrainerMedNext):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerMedNext_100epochs(nnUNetTrainerMedNext):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerMedNext_250epochs(nnUNetTrainerMedNext):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerMedNext_500epochs(nnUNetTrainerMedNext):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerMedNext_1000epochs(nnUNetTrainerMedNext):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


class nnUNetTrainerMedNext_1250epochs(nnUNetTrainerMedNext):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1250


class nnUNetTrainerMedNext_1250epochs_BS1(nnUNetTrainerMedNext_1250epochs):
    """
    MedNeXt-M k3 for 1250 epochs with batch size pinned to 1, so the real training
    step (~13.8 GB at 128^3) fits comfortably on a 24 GB GPU (BS2 peaks ~22.7 GB and
    OOMs). Batch size is forced here because the nnUNetPlans 3d_fullres config uses 2.
    """
    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample()
        if not self.is_ddp:
            self.batch_size = 1


# ── Warm restart variants (M, k=3) ──────────────────────────────────────


class nnUNetTrainerMedNext_warmRestart(nnUNetTrainerMedNext):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        return optimizer, lr_scheduler


class nnUNetTrainerMedNext_50epochs_warmRestart(nnUNetTrainerMedNext_warmRestart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerMedNext_100epochs_warmRestart(nnUNetTrainerMedNext_warmRestart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerMedNext_250epochs_warmRestart(nnUNetTrainerMedNext_warmRestart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerMedNext_500epochs_warmRestart(nnUNetTrainerMedNext_warmRestart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerMedNext_1000epochs_warmRestart(nnUNetTrainerMedNext_warmRestart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
