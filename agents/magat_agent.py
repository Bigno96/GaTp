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
import timeit

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from typing import Optional
from statistics import mean

from agents.base_agent import Agent
from data_loading.data_loader import GaTpDataLoader
from models.magat_net import MAGATNet
from utils.multi_agent_simulator import MultiAgentSimulator
from utils.metrics import Performance, PerformanceRecorder


class MagatAgent(Agent):

    def __init__(self, config):
        super(MagatAgent, self).__init__(config)

        # initialize data loader
        self.data_loader: GaTpDataLoader = GaTpDataLoader(config=self.config)

        # initialize model
        self.model: torch.nn.Module = MAGATNet(config=self.config)
        self.logger.info(f'MAGAT Model: {self.model}\n')

        # define loss
        self.loss = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        # define optimizers
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)  # L2 regularize
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                              T_max=self.config.max_epoch,  # max number of iteration
                                                              eta_min=1e-6)     # min learning rate

        # initialize counters
        self.current_epoch = 0
        self.performance: Performance = Performance()
        self.time_record = 0.0

        # set cuda flag
        self.cuda: bool = torch.cuda.is_available()   # check availability
        if self.cuda and not self.config.cuda:  # user has cuda available, but not enabled
            self.logger.info('WARNING: You have a CUDA device, you should probably enable CUDA')

        self.cuda = self.cuda and self.config.cuda  # prevent setting cuda True if not available

        # set the manual seed for torch
        self.manual_seed: int = self.config.seed
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
        if self.config.sim_num_process == 0:    # single process
            self.simulate_agent_exec = self.sim_agent_exec_single
        else:
            self.simulate_agent_exec = self.sim_agent_exec_single   # TODO

        # load checkpoint if necessary
        if config.load_checkpoint:
            self.load_checkpoint(epoch=config.load_epoch,
                                 best=config.load_ckp_mode == 'best',
                                 latest=config.load_ckp_mode == 'latest')

        # simulation handling classes and variables
        self.simulator: MultiAgentSimulator = MultiAgentSimulator(config=self.config)
        self.recorder: PerformanceRecorder = PerformanceRecorder(simulator=self.simulator)
        self.best_performance: Performance = Performance()

    def save_checkpoint(self, epoch=0, is_best=False, latest=True):
        """
        Checkpoint saver
        :param epoch: int, current epoch being saved
        :param is_best: bool, flag to indicate whether current checkpoint's metric is the best so far
        :param latest: bool, flag to indicate the checkpoint is the latest one trained
        """
        if latest:
            file_name = 'checkpoint.pth.tar'    # latest checkpoint -> unnamed
        else:
            file_name = f'checkpoint_{epoch:03d}.pth.tar'   # name checkpoint

        state = {
            'epoch': self.current_epoch + 1,    # next epoch is saved, since this one is finished
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

    def load_checkpoint(self, epoch=0, best=False, latest=True):
        """
        Checkpoint loader
        Priority: latest -> best -> epoch
        :param epoch: int, current epoch being loaded
        :param best: bool, flag to indicate whether loading best checkpoint or not
        :param latest: bool, flag to indicate the loaded checkpoint is the latest one trained
        """
        # order of priority: latest -> best -> specific epoch
        if latest:
            file_name = 'checkpoint.pth.tar'
        elif best:
            file_name = 'model_best.pth.tar'
        else:
            file_name = f'checkpoint_{epoch:03d}.pth.tar'

        file_path = os.path.join(self.config.checkpoint_dir, file_name)
        try:
            self.logger.info(f'Loading checkpoint "{file_name}"')
            # load checkpoint, moving tensors onto selected device (cuda or cpu)
            checkpoint = torch.load(f=file_path,
                                    map_location=torch.device(f'{self.config.device}'))

            # load back parameters
            self.current_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.logger.info(f'Checkpoint loaded successfully from "{self.config.checkpoint_dir}"'
                             f'at (epoch {checkpoint["epoch"]}) at (iteration {checkpoint["iteration"]})\n')
        # no file found
        except OSError:
            self.logger.info(f'No checkpoint exists from "{self.config.checkpoint_dir}". Skipping.')

    def run(self):
        """
        The main operator
        """
        try:
            # testing mode
            if self.config.mode == 'test':
                start_time = timeit.default_timer()
                self.test()
                self.time_record = timeit.default_timer() - start_time
            # training mode
            else:
                self.train()

        # interrupting training or testing by keyboard
        except KeyboardInterrupt:
            self.logger.info("Entered CTRL+C. Wait to finalize")

    def train(self):
        """
        Main training loop
        """
        self.logger.info('Start training')
        # loop over epochs 
        # start from current_epoch -> in case of loaded checkpoint
        for epoch in range(self.current_epoch, self.config.max_epoch+1):
            self.current_epoch = epoch      # update epoch
            self.logger.info(f'Begin Epoch {self.current_epoch} - Learning Rate: {self.scheduler.get_last_lr()}')

            self.train_one_epoch()          # train the epoch

            # set to None to avoid useless instancing
            performance: Optional[Performance] = None
            # always validate first 4 epochs
            if epoch <= 4:
                performance = self.validate()
                self.save_checkpoint(epoch, is_best=False, latest=False)
            # else validate only every n epochs
            elif epoch % self.config.validate_every == 0:
                performance = self.validate()
                self.save_checkpoint(epoch, is_best=False, latest=False)

            # if performance was instanced
            if performance:
                is_best = performance > self.best_performance   # check if it is the best one
                if is_best:     # if so
                    # save performance value and best checkpoint
                    self.best_performance = performance
                    self.save_checkpoint(epoch=epoch, is_best=is_best, latest=True)

            self.scheduler.step()

    def validate(self) -> Performance:
        """
        Validate current model
        :return: mean performance recorder during the validation simulation
        """
        self.logger.info('Start validation')
        data_loader = self.data_loader.valid_loader     # get valid loader
        # return mean performance of the simulation
        return self.simulate_agent_exec(data_loader=data_loader)

    def test(self):
        """
        Main testing loop
        """
        self.logger.info('Start testing')
        data_loader = self.data_loader.valid_loader  # get test loader
        # simulate
        mean_performance = self.simulate_agent_exec(data_loader=data_loader)
        # set mean performance for printing
        self.best_performance = mean_performance

    def finalize(self):
        """
        Concluding all operations and printing results
        """
        if self.config.mode == 'test':
            print("################## End of testing ################## ")
            print(f'Testing mean performance:\t{self.best_performance}')
            print(f'Computation time:\t{self.time_record} ')
        # train mode
        else:
            print("################## End of training ################## ")
            print(f'Best Validation performance:\t{self.best_performance}')

    def train_one_epoch(self):
        """
        One epoch of training
        """

        # set the model to be in training mode
        self.model.train()

        # reference this for warning on final print outside the loop
        loss = 0

        # loop over various batches of training data
        for batch_idx, (batch_input, batch_GSO, batch_target) \
                in enumerate(self.data_loader.train_loader):

            # move all tensors to the correct device
            # batch x agent_num x channel_num x FOV+2*border x FOV+2*border
            batch_input = batch_input.to(self.config.device)
            # batch x agent_num x agent_num
            batch_GSO = batch_GSO.to(self.config.device)
            # batch x agent_num x 5
            batch_target = batch_target.to(self.config.device)

            # B -> batch size
            # N -> agent number
            # C -> input channels
            # H, W -> height and width of the input channels
            B, N, _, _, _ = batch_input.shape

            # reshape for compatibility with model output
            batch_target = batch_target.reshape(B * N, 5)

            # init model, optimizer and loss
            self.model.set_gso(batch_GSO)
            self.optimizer.zero_grad()
            loss = 0

            # get model prediction, B*N x 5
            predict = self.model(batch_input)

            # compute loss
            # torch.max returns both values and indices
            # torch.max axis = 1 -> find the index of the chosen action for each agent
            loss = loss + self.loss(predict, torch.max(batch_target, 1)[1])     # [1] to unpack indices

            # update gradient with backward pass
            loss.backward()
            self.optimizer.step()

            # log progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(f'Epoch {self.current_epoch}:'
                                 f'[{batch_idx * len(batch_input)}/{len(self.data_loader.train_loader.dataset)}'
                                 f'({100 * batch_idx / len(self.data_loader.train_loader):.0f}%)]\t'
                                 f'Loss: {loss.item():.6f}')
        # always last batch logged
        self.logger.info(f'Epoch {self.current_epoch}:'
                         f'[{len(self.data_loader.train_loader.dataset)}/{len(self.data_loader.train_loader.dataset)}'
                         f'({100.:.0f}%)]\t'
                         f'Loss: {loss.item():.6f}')

    def sim_agent_exec_single(self, data_loader: data.DataLoader) -> Performance:
        """
        Simulate all MAPD problem in the data loader using trained model for solving it
        All the simulations are done in a single process, sequentially
        :param data_loader: pytorch Dataloader, either testing or valid DataLoader
        :return mean performance recorder during the validation simulation
        """
        self.logger.info('Starting MAPD simulations')

        performance_list: list[Performance] = []
        with torch.no_grad():
            # loop over all cases in the test/valid data loader
            for case_idx, (obstacle_map, start_pos_list, task_list, makespan, service_time) \
                    in enumerate(data_loader):
                # simulate the MAPD execution
                self.simulator.simulate(obstacle_map=obstacle_map,
                                        start_pos_list=start_pos_list,
                                        task_list=task_list,
                                        model=self.model,
                                        target_makespan=makespan.item())

                # collect metrics
                performance = self.recorder.evaluate_performance(target_makespan=makespan.item())
                performance_list.append(performance)

                # if testing, update at each simulation
                if self.config.mode == 'test':
                    self.logger.info(f'Case {case_idx+1}: [{case_idx+1}/{len(data_loader)}'
                                     f'({100 * (case_idx+1) / len(data_loader):.0f}%)]\t'
                                     f'{performance}')
                # else, validation, update every 50 sim
                else:
                    if case_idx % 50 == 0:
                        self.logger.info(f'Case {case_idx}: [{case_idx}/{len(data_loader)}'
                                         f'({100 * case_idx / len(data_loader):.0f}%)]\t'
                                         f'{performance}')

        # collect all the metrics
        compl_task = [p.completed_task for p in performance_list]
        coll = [p.collisions_difference for p in performance_list]
        mks = [p.makespan_difference for p in performance_list]

        # return a Performance instance
        return Performance(completed_task=mean(compl_task),
                           collisions_difference=mean(coll),
                           makespan_difference=mean(mks))

    def sim_agent_exec_multi(self):
        # TODO
        pass
