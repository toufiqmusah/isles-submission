from nnunetv2.training.nnUNetTrainer.hff.nnUNetTrainerHFF import nnUNetTrainerHFF


class nnUNetTrainerHFF_1200(nnUNetTrainerHFF):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
