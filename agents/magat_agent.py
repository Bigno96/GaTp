"""
Codes for running training and evaluation of Decentralised Path Planning with Message-Aware Graph Attention Networks.

The model was proposed by us in the below paper:
Q. Li, W. Lin, Z. Liu and A. Prorok,
"Message-Aware Graph Attention Networks for Large-Scale Multi-Robot Path Planning"
in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 5533-5540, July 2021, doi: 10.1109/LRA.2021.3077863.
"""

import shutil
import torch
import os

import torch.nn as nn
import torch.optim as optim

from agents.base_agent import Agent
from data_loading.data_loader import GaTpDataLoader
from models.magat_net import MAGATNet


class MagatAgent(Agent):

    def __init__(self, config):
        super(MagatAgent, self).__init__(config)

        # initialize data loader
        self.data_loader = GaTpDataLoader(config=self.config)

        # initialize model
        self.model = MAGATNet(config=self.config)
        self.logger.info(f'MAGAT Model: {self.model}\n')

        # define loss
        self.loss = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        # define optimizers
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)      # L2 regularize
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                              T_max=self.config.max_epoch,  # max number of iteration
                                                              eta_min=1e-6)     # min learning rate

        # initialize counters
        self.current_epoch = 0
        self.current_iteration = 0

        # set cuda flag
        self.cuda = torch.cuda.is_available()       # check availability
        if self.cuda and not self.config.cuda:      # user has cuda available, but not enabled
            self.logger.info('WARNING: You have a CUDA device, you should probably enable CUDA')

        self.cuda = self.cuda & self.config.cuda    # prevent setting cuda True if not available

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        # cuda enabled
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)

            self.config.device = torch.device(f'cuda:{self.config.gpu_device}')
            torch.cuda.set_device(self.config.gpu_device)

            self.model = self.model.to(self.config.device)
            self.loss = self.loss.to(self.config.device)
            self.logger.info('Program will run on ***GPU-CUDA***\n')
        # cpu is used
        else:
            self.config.device = torch.device('cpu')
            torch.manual_seed(self.manual_seed)
            self.logger.info('Program will run on ***CPU***\n')

        # set agent simulation function
        if self.config.sim_num_process == 0:        # single process
            print('Using single thread for agent simulation')
            self.simulate_agent_exec = self.sim_agent_exec_single
        else:
            print(f'Using multi threads for agent simulation'
                  f'Thread num: {self.config.sim_num_process}')
            self.simulate_agent_exec = self.sim_agent_exec_multi

    def save_checkpoint(self, epoch, is_best=False, latest=True):
        """
        Checkpoint saver
        :param epoch: int, current epoch being saved
        :param is_best: bool, flag to indicate whether current checkpoint's metric is the best so far
        :param latest: bool, flag to indicate the checkpoint is the latest one trained
        """
        if latest:
            file_name = 'checkpoint.pth.tar'        # latest checkpoint -> unnamed
        else:
            file_name = f'checkpoint_{epoch:03d}.pth.tar'   # name checkpoint

        state = {
            'epoch': self.current_epoch + 1,        # next epoch is saved, since this one is finished
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        # save the state
        torch.save(state, os.path.join(self.config.checkpoint_dir, file_name))
        # if it is the best, copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(os.path.join(self.config.checkpoint_dir, file_name),
                            os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))

    def load_checkpoint(self, epoch, best=False, latest=True):
        """
        Checkpoint loader
        Priority: latest -> best -> epoch
        :param epoch: int, current epoch being loaded
        :param best: bool, flag to indicate whether loading best checkpoint or not
        :param latest: bool, flag to indicate the loaded checkpoint is the latest one trained
        """
        # order of priority: latest -> best -> specific epoch
        if latest:
            file_name = "checkpoint.pth.tar"
        elif best:
            file_name = "model_best.pth.tar"
        else:
            file_name = "checkpoint_{:03d}.pth.tar".format(epoch)

        file_path = os.path.join(self.config.checkpoint_dir, file_name)
        try:
            self.logger.info(f'Loading checkpoint "{file_name}"')
            # load checkpoint, moving tensors onto selected device (cuda or cpu)
            checkpoint = torch.load(f=file_path,
                                    map_location=torch.device(f'{self.config.device}'))

            # load back parameters
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.logger.info(f'Checkpoint loaded successfully from "{self.config.checkpoint_dir}"'
                             f'at (epoch {checkpoint["epoch"]}) at (iteration {checkpoint["iteration"]})\n')
        # no file found
        except OSError:
            self.logger.info(f'No checkpoint exists from "{self.config.checkpoint_dir}". Skipping.')

    def run(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass

    def finalize(self):
        pass

    def sim_agent_exec_single(self):
        pass

    def sim_agent_exec_multi(self):
        pass
