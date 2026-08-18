import torch

from nnunetv2.training.nnUNetTrainer.hff.nnUNetTrainerHFF import nnUNetTrainerHFF
from nnunetv2.training.lr_scheduler.warmup import PolyLRScheduler_warm_restarts


class nnUNetTrainerHFF_500_WarmRestarts(nnUNetTrainerHFF):
    """HFF trainer for 500 epochs with PolyLR warm restarts every 100 epochs (5 cycles)."""

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.restart_period = 100  # warm restart every 100 epochs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler_warm_restarts(
            optimizer, self.initial_lr, self.num_epochs,
            start_step=0,
            restart_period=self.restart_period,
        )
        self.print_to_log_file(
            f"Initialized SGD optimizer and PolyLR warm restarts scheduler "
            f"(restart every {self.restart_period} epochs) at epoch {self.current_epoch}"
        )
        return optimizer, lr_scheduler
