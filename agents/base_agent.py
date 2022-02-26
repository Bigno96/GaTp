"""
Base Agent class, where all other agents inherit from
Contains definitions for all the necessary functions
"""

import logging


class Agent:
    """
    This base class will contain the base functions to be overloaded by any agent implemented
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def save_checkpoint(self, epoch, is_best=False, latest=True):
        """
        Checkpoint saver
        :param epoch: current epoch being saved
        :param is_best: flag to indicate whether current checkpoint's metric is the best so far
        :param latest: flag to indicate the checkpoint is the latest one trained
        """
        raise NotImplementedError

    def load_checkpoint(self, epoch, best=False, latest=True):
        """
        Checkpoint loader
        :param epoch: int, current epoch being loaded
        :param best: bool, flag to indicate whether loading best checkpoint or not
        :param latest: bool, flag to indicate the loaded checkpoint is the latest one trained
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        """
        raise NotImplementedError

    def validate(self):
        """
        Model validation
        """
        raise NotImplementedError

    def test(self):
        """
        Model testing
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        """
        raise NotImplementedError
