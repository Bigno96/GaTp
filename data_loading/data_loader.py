"""
PyTorch Custom Data Loader implementation
"""

import logging
import data_loading.dataset as dataset

from torch.utils.data import DataLoader
from easydict import EasyDict


class GaTpDataLoader:
    """
    Custom data loader for GaTp model
    """

    def __init__(self,
                 config: EasyDict):
        """
        :param config: Namespace of dataset configurations
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")

        # data loader for training
        if self.config.mode == 'train':

            self.train_dataset = dataset.GaTpDataset(self.config, 'train')
            self.valid_dataset = dataset.GaTpDataset(self.config, 'valid')

            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.batch_size,
                                           shuffle=self.config.shuffle_train,   # usually True
                                           num_workers=self.config.data_loader_workers,  # num of processes
                                           pin_memory=self.config.pin_memory)

            self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                           batch_size=self.config.valid_batch_size,
                                           shuffle=False,   # don't shuffle valid set
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)

        # # data loader for validation only mode
        elif self.config.mode == 'valid':

            self.valid_dataset = dataset.GaTpDataset(self.config, 'valid')

            self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                           batch_size=self.config.valid_batch_size,
                                           shuffle=False,  # don't shuffle valid set
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)

        # data loader for testing
        elif self.config.mode == 'test':

            self.test_dataset = dataset.GaTpDataset(self.config, 'test')

            self.test_loader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.config.test_batch_size,
                                          shuffle=False,    # don't shuffle valid set
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)

        else:
            self.logger.error('Incorrect operating mode was specified')
