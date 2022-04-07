"""
Base Agent class, where all other agents inherit from
Contains definitions for all the necessary functions
"""

import logging

import utils.metrics as metrics


class Agent:
    """
    This base class will contain the base functions to be overloaded by any agent implemented
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('Agent')

    def save_checkpoint(self,
                        best: bool = False,
                        latest: bool = True,
                        epoch: int = 0
                        ) -> None:
        """
        Checkpoint saver
        :param epoch: current epoch being saved
        :param best: flag to indicate whether current checkpoint's metric is the best so far
        :param latest: flag to indicate the checkpoint is the latest one trained
        """
        raise NotImplementedError

    def load_checkpoint(self,
                        best: bool = False,
                        latest: bool = True,
                        epoch: int = 0
                        ) -> None:
        """
        Checkpoint loader
        :param epoch: int, current epoch being loaded
        :param best: bool, flag to indicate whether loading best checkpoint or not
        :param latest: bool, flag to indicate the loaded checkpoint is the latest one trained
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        The main operator
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Main training loop
        """
        raise NotImplementedError

    def validate(self,
                 checkpoint_path: str
                 ) -> metrics.Performance:
        """
        Model validation
        :param checkpoint_path: path to the checkpoint to load model from
        """
        raise NotImplementedError

    def test(self) -> None:
        """
        Model testing
        """
        raise NotImplementedError

    def finalize(self) -> None:
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        """
        raise NotImplementedError
