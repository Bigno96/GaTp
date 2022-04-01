"""
Codes for running training and evaluation of Decentralised Path Planning with Message-Aware Graph Attention Networks.

The model was proposed by us in the below paper:
Q. Li, W. Lin, Z. Liu and A. Prorok,
"Message-Aware Graph Attention Networks for Large-Scale Multi-Robot Path Planning"
in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 5533-5540, July 2021, doi: 10.1109/LRA.2021.3077863.

Cyclic Learning Rate as proposed in:
L.N. Smith,
"Cyclical learning rates for training neural networks"
In 2017 IEEE winter conference on applications of computer vision (WACV), pp. 464-472.
"""

import shutil
import time
import torch
import os
import timeit
import logging

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import agents.base_agent as agents
import data_loading.data_loader as loader
import models.magat_net as magat
import utils.multi_agent_simulator as sim
import utils.metrics as metrics
import torch.multiprocessing as mp

from easydict import EasyDict
from torchinfo import summary
from torch.multiprocessing.queue import SimpleQueue
from typing import List


class MagatAgent(agents.Agent):

    def __init__(self,
                 config: EasyDict):
        super(MagatAgent, self).__init__(config)

        # initialize counters
        self.current_epoch = 0
        self.performance = metrics.Performance()
        self.best_performance = metrics.Performance()
        self.time_record = 0.0

        # set cuda flag
        self.cuda: bool = torch.cuda.is_available()   # check availability
        if self.cuda and not self.config.cuda:  # user has cuda available, but not enabled
            self.logger.info('WARNING: You have a CUDA device, you should probably enable CUDA')
        if not self.cuda and self.config.cuda:  # user has selected cuda, but it is not available
            self.logger.info(f'WARNING: You have selected CUDA device, but no available CUDA device was found\n'
                             f'Switching to CPU instead')
        self.cuda = self.cuda and self.config.cuda  # prevent setting cuda True if not available

        # set the manual seed for torch
        self.manual_seed: int = self.config.seed

        # set up device
        self.setup_device()

        # scaler for AMP acceleration
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cuda)

        # initialize data loader
        self.data_loader = loader.GaTpDataLoader(config=self.config)

        # initialize the model
        self.model = magat.MAGATNet(config=self.config).to(self.config.device)

        # define loss
        self.loss_f = nn.CrossEntropyLoss().to(self.config.device)

        # define optimizer
        self.optimizer = optim.AdamW(params=self.model.parameters(),
                                     lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)  # L2 regularize

        # define scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                       max_lr=self.config.learning_rate*10,
                                                       epochs=self.config.max_epoch,
                                                       steps_per_epoch=len(self.data_loader.train_loader))

        # set agent simulation function
        if self.config.sim_num_process <= 1 or os.name == 'nt':  # single process or Windows
            # pytorch does not support sharing GPU resources between processes on Windows
            self.simulate_agent_exec = self.sim_agent_exec_single
        else:
            self.simulate_agent_exec = self.sim_agent_exec_multi

        # simulation handling classes and variables
        self.simulator = sim.MultiAgentSimulator(config=self.config)    # needs config.device set in setup_device()
        self.recorder = metrics.PerformanceRecorder(simulator=self.simulator)

        # load checkpoint if necessary
        if config.load_checkpoint:
            self.load_checkpoint(epoch=config.epoch_id,
                                 best=config.load_ckp_mode == 'best',
                                 latest=config.load_ckp_mode == 'latest')
            # index of the last batch, used when resuming a training job
            # this number represents the total number of batches computed,
            # not the total number of epochs computed
            self.scheduler.last_epoch = self.current_epoch

        # use cuDNN benchmarking
        if self.cuda:
            torch.backends.cudnn.benchmark = True

        '''print summary of the model'''
        batch_size = self.config.batch_size
        agent_num = self.config.agent_number
        channel_num = 3
        H, W = self.config.FOV + 2, self.config.FOV + 2
        dummy_GSO = torch.ones(size=(batch_size, agent_num, agent_num),
                               device=self.config.device,
                               dtype=torch.float)
        self.model.set_gso(dummy_GSO)
        summary(model=self.model,
                input_size=(batch_size, agent_num, channel_num, H, W),
                device=self.config.device,
                col_names=["input_size", "output_size", "num_params"])

    def setup_device(self) -> None:
        """
        Move model and losses accordingly to chosen device
        Set also random seed
        """
        # cuda enabled
        if self.cuda:
            self.config.device = torch.device(f'cuda:{self.config.gpu_device}')
            torch.cuda.set_device(self.config.gpu_device)

            torch.cuda.manual_seed_all(self.manual_seed)
            self.logger.info('Program will run on ***GPU-CUDA***\n')

        # cpu is used
        else:
            self.config.device = torch.device('cpu')

            torch.manual_seed(self.manual_seed)
            self.logger.info('Program will run on ***CPU***\n')

    def save_checkpoint(self,
                        epoch: int = 0,
                        is_best: bool = False,
                        latest: bool = True
                        ) -> None:
        """
        Checkpoint saver
        :param epoch: current epoch being saved
        :param is_best: flag to indicate whether current checkpoint's metric is the best so far
        :param latest: flag to indicate the checkpoint is the latest one trained
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

    def load_checkpoint(self,
                        epoch: int = 0,
                        best: bool = False,
                        latest: bool = True
                        ) -> None:
        """
        Checkpoint loader
        Priority: latest -> best -> epoch
        :param epoch: current epoch being loaded
        :param best: flag to indicate whether loading best checkpoint or not
        :param latest: flag to indicate the loaded checkpoint is the latest one trained
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
                             f'at epoch {checkpoint["epoch"]}\n')
        # no file found
        except OSError:
            self.logger.info(f'No checkpoint exists from "{self.config.checkpoint_dir}". Skipping.')

    def run(self) -> None:
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

    def train(self) -> None:
        """
        Main training loop
        """
        self.logger.info('Start training')
        # loop over epochs 
        # start from current_epoch -> in case of loaded checkpoint
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch      # update epoch
            self.logger.info(f'Begin Epoch {self.current_epoch}')

            self.train_one_epoch()  # train the epoch

            # validate only every n epochs
            if epoch % self.config.validate_every == 0:
                self.performance = self.validate()
                self.save_checkpoint(epoch=epoch, is_best=False, latest=False)
                self.logger.info(f'Validation {self.performance}')

            # check if it is the best one
            is_best = self.performance > self.best_performance
            if is_best:     # if so
                # save performance value and best checkpoint
                self.best_performance = self.performance.copy()
                self.save_checkpoint(epoch=epoch, is_best=True, latest=True)

    def validate(self) -> metrics.Performance:
        """
        Validate current model
        :return: mean performance recorder during the validation simulation
        """
        self.logger.info('Start validation')
        data_loader = self.data_loader.valid_loader     # get valid loader
        # return mean performance of the simulation
        return self.simulate_agent_exec(data_loader=data_loader)

    def test(self) -> None:
        """
        Main testing loop
        """
        self.logger.info('Start testing')
        data_loader = self.data_loader.test_loader  # get test loader
        # simulate
        mean_performance = self.simulate_agent_exec(data_loader=data_loader)
        # set mean performance for printing
        self.best_performance = mean_performance

    def finalize(self) -> None:
        """
        Concluding all operations and printing results
        """
        if self.config.mode == 'test':
            print("################## End of testing ################## ")
            print(f'Testing Mean {self.best_performance}')
            print(f'Computation time:\t{self.time_record} ')
        # train mode
        else:
            print("################## End of training ################## ")
            print(f'Best Validation {self.best_performance}')

    def train_one_epoch(self) -> None:
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

            # init model and loss
            self.model.set_gso(batch_GSO)
            loss = 0

            # AMP optimization
            with torch.cuda.amp.autocast(enabled=self.cuda):
                # get model prediction, B*N x 5
                predict = self.model(batch_input)
                # compute loss
                # torch.max returns both values and indices
                # torch.max axis = 1 -> find the index of the chosen action for each agent
                loss = loss + self.loss_f(predict, torch.max(batch_target, 1)[1])  # [1] to unpack indices

            # update gradient with backward pass using AMP scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.step()

            # scheduler step, cyclic lr scheduling -> step after each batch
            self.scheduler.step()

            # log progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(f'Epoch {self.current_epoch}:'
                                 f'[{batch_idx * self.config.batch_size}/{len(self.data_loader.train_loader.dataset)}'
                                 f'({100 * batch_idx / len(self.data_loader.train_loader):.0f}%)] '
                                 f'- Learning Rate: {self.scheduler.get_last_lr()}\t'
                                 f'Loss: {loss.item():.6f}')

        # always last batch logged
        self.logger.info(f'Epoch {self.current_epoch}:'
                         f'[{len(self.data_loader.train_loader.dataset)}/{len(self.data_loader.train_loader.dataset)}'
                         f'({100.:.0f}%)]\t'
                         f'Loss: {loss.item():.6f}')

    def sim_agent_exec_single(self,
                              data_loader: data.DataLoader
                              ) -> metrics.Performance:
        """
        Simulate all MAPD problem in the data loader using trained model for solving it
        All the simulations are done in a single process, sequentially
        :param data_loader: pytorch Dataloader, either testing or valid DataLoader
        :return mean performance recorder during the validation simulation
        """
        self.logger.info('Starting single process MAPD simulations')

        performance_list: List[metrics.Performance] = []
        with torch.no_grad():
            # loop over all cases in the test/valid data loader
            for case_idx, (obstacle_map, start_pos_list, task_list, makespan, service_time) \
                    in enumerate(data_loader):
                # simulate the MAPD execution
                # batch size = 1 -> unpack all tensors
                self.simulator.simulate(obstacle_map=obstacle_map[0],
                                        start_pos_list=start_pos_list[0],
                                        task_list=task_list[0],
                                        model=self.model,
                                        target_makespan=makespan.item())

                # collect metrics
                performance = self.recorder.evaluate_performance(target_makespan=makespan.item())
                performance_list.append(performance)
                metrics.print_performance(performance=performance,
                                          mode=self.config.mode,
                                          logger=self.logger,
                                          case_idx=case_idx,
                                          max_size=len(data_loader))

        # return average performances
        return metrics.get_avg_performance(performance_list=performance_list)

    def sim_agent_exec_multi(self,
                             data_loader: data.DataLoader
                             ) -> metrics.Performance:
        """
        Simulate all MAPD problem in the data loader using trained model for solving it
        Multiprocessing simulation
        :param data_loader: pytorch Dataloader, either testing or valid DataLoader
        :return mean performance recorder during the validation simulation
        """
        self.logger.info('Starting multi process MAPD simulations')

        # useful variables
        data_size = len(data_loader)

        # load data from data loader
        with torch.no_grad():
            # set up queues
            performance_queue = SimpleQueue(ctx=mp.get_context())
            data_queue = SimpleQueue(ctx=mp.get_context())
            for i, data_ in enumerate(data_loader):
                data_queue.put((i, data_))
            # collect args for multiprocessing
            # model, simulator, recorder, data queue, mode, logger, performance queue, data size
            args = (self.model,
                    self.simulator,
                    self.recorder,
                    data_queue,
                    self.config.mode,
                    self.logger,
                    performance_queue,
                    data_size)

            # spawn and run processes, wait all to finish
            mp.spawn(fn=sim_worker,
                     args=args,
                     nprocs=self.config.sim_num_process,
                     join=False)

            # release data queue
            data_queue.close()

            # get performance list
            performance_queue.put('STOP')   # termination sentinel
            performance_list = [p for p in iter(performance_queue.get, 'STOP')]
            time.sleep(.1)      # release the GIL

            # release performance queue
            performance_queue.close()

        # return average performances
        # noinspection PyTypeChecker
        return metrics.get_avg_performance(performance_list=performance_list)


# noinspection PyUnusedLocal
def sim_worker(process_id: int,     # needed because of spawn implementation
               model: nn.Module,
               simulator: sim.MultiAgentSimulator,
               recorder: metrics.PerformanceRecorder,
               data_queue: SimpleQueue,
               mode: str,
               logger: logging.Logger,
               performance_queue: SimpleQueue,
               data_size: int,
               ) -> None:
    """
    Class for multiprocessing simulation
    Simulate MAPD execution and record performances over input data taken from iterating over a dataloader
    :param process_id: id of the process that executes the function, automatically passed by mp spawn
    :param model: model used to simulate movements
    :param simulator: instance of multiagent simulator to carry on the simulation
    :param recorder: used to compute performance of a simulation
    :param data_queue: contains shared input data to run simulation over
    :param mode: 'test' or 'train'
    :param logger: logger used to print the info on
    :param performance_queue: shared queue to put simulation results
    :param data_size: total length of the dataset, needed for print purposes
    """
    with torch.no_grad():
        while not data_queue.empty():
            # unpack tensors from input data queue
            case_idx, (obstacle_map, start_pos_list,
                       task_list, makespan, service_time) = data_queue.get()

            # simulate the MAPD execution
            # batch size = 1 -> unpack all tensors
            simulator.simulate(obstacle_map=obstacle_map[0],
                               start_pos_list=start_pos_list[0],
                               task_list=task_list[0],
                               model=model,
                               target_makespan=makespan.item())

            # collect metrics
            performance = recorder.evaluate_performance(target_makespan=makespan.item())
            # print metrics
            metrics.print_performance(performance=performance,
                                      mode=mode,
                                      logger=logger,
                                      case_idx=case_idx,
                                      max_size=data_size)

            # add metrics
            performance_queue.put(performance)
