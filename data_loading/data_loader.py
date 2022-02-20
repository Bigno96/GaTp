"""
PyTorch Custom Data Loader implementation
"""

from torch.utils.data import DataLoader
from data_loading.dataset import GaTpDataset


class GaTpDataLoader:
    """
    Custom data loader for GaTp model
    """

    def __init__(self, config):
        """
        :param config: Namespace of dataset configurations
        """
        self.config = config
        assert self.config.mode in ['train', 'test']

        # data loader for training
        if self.config.mode == 'train':

            self.train_dataset = GaTpDataset(self.config, 'train')
            self.valid_dataset = GaTpDataset(self.config, 'valid')

            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.batch_size,
                                           shuffle=True,
                                           num_workers=self.config.data_loader_workers,     # num of processes
                                           pin_memory=self.config.pin_memory)

            self.valid_loader = DataLoader(self.valid_dataset,
                                           batch_size=self.config.valid_batch_size,
                                           shuffle=False,               # don't shuffle valid set
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)

        # data loader for testing
        else:
            self.test_dataset = GaTpDataset(self.config, 'test')

            self.test_loader = DataLoader(self.test_dataset,
                                          batch_size=self.config.test_batch_size,
                                          shuffle=False,                # don't shuffle valid set
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)
