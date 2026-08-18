from monai.networks.nets import SegResNet as SegResNet_
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)


class nnUNetTrainerSegResNet(nnUNetTrainer):
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
        self.enable_deep_supervision = False
        self.save_every = 100

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        spatial_dims = len(configuration_manager.patch_size)

        # Cap stages at 4 to avoid spatial dimension mismatches between encoder
        # skip connections and decoder upsamplings when patch dims become odd.
        # SegResNet default: 4 down, 3 up.
        n_stages = min(len(configuration_manager.pool_op_kernel_sizes), 4)

        blocks_down = (
            tuple([1] + [2] * (n_stages - 2) + [4])
            if n_stages >= 3
            else tuple([1] * n_stages)
        )
        blocks_up = tuple([1] * (n_stages - 1))

        return SegResNet_(
            spatial_dims=spatial_dims,
            init_filters=32,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            dropout_prob=None,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode="deconv",
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _do_i_compile(self):
        return False


class nnUNetTrainerSegResNet_50epochs(nnUNetTrainerSegResNet):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerSegResNet_100epochs(nnUNetTrainerSegResNet):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerSegResNet_250epochs(nnUNetTrainerSegResNet):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerSegResNet_500epochs(nnUNetTrainerSegResNet):
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


class nnUNetTrainerSegResNet_1000epochs(nnUNetTrainerSegResNet):
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


class nnUNetTrainerSegResNet_warmRestart(nnUNetTrainerSegResNet):
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

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
        return optimizer, lr_scheduler


class nnUNetTrainerSegResNet_50epochs_warmRestart(nnUNetTrainerSegResNet_warmRestart):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerSegResNet_100epochs_warmRestart(nnUNetTrainerSegResNet_warmRestart):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerSegResNet_250epochs_warmRestart(nnUNetTrainerSegResNet_warmRestart):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerSegResNet_500epochs_warmRestart(nnUNetTrainerSegResNet_warmRestart):
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


class nnUNetTrainerSegResNet_1000epochs_warmRestart(nnUNetTrainerSegResNet_warmRestart):
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
