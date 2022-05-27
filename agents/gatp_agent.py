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
import torch
import os
import timeit
import shutil
import pickle

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import agents.base_agent as agents
import data_loading.data_loader as loader
import models.gatp_net as gatp
import utils.multi_agent_simulator as sim
import utils.metrics as metrics
import torch.multiprocessing as mp
import utils.aggregator as agg

from easydict import EasyDict
from pytorch_model_summary import summary
from torch.multiprocessing.queue import Queue
from typing import List

MIN_RL = 1e-6
STOP_SENTINEL = 'STOP'


class GaTpAgent(agents.Agent):

    def __init__(self,
                 config: EasyDict):
        super(GaTpAgent, self).__init__(config)

        # initialize counters
        self.current_epoch = 0
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
        # set amp flag
        self.amp = self.config.amp and self.cuda    # no amp if cpu is used
        # set cuDNN benchmarking flag
        torch.backends.cudnn.benchmark = self.config.cuda_benchmark and self.cuda

        # set the manual seed for torch
        self.manual_seed: int = self.config.seed
        # set up device
        self.setup_device()

        # initialize data loader
        self.data_loader = loader.GaTpDataLoader(config=self.config)

        # set up training variables
        if self.config.mode == 'train':
            # scaler for AMP acceleration
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

            # initialize the model
            self.model = gatp.GaTpNet(config=self.config).to(self.config.device)

            # define loss
            self.loss_f = nn.CrossEntropyLoss().to(self.config.device)

            # define optimizer
            self.optimizer = optim.AdamW(params=self.model.parameters(),
                                         lr=self.config.learning_rate,
                                         weight_decay=self.config.weight_decay)

            # define scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                  T_max=self.config.max_epoch,
                                                                  eta_min=MIN_RL)

        # set agent simulation function
        if self.config.sim_num_process <= 1:
            self.simulate_agent_exec = self.sim_agent_exec_single
        else:
            self.simulate_agent_exec = self.sim_agent_exec_multi

        # load checkpoint if necessary
        if config.load_checkpoint:
            self.load_checkpoint(epoch=config.epoch_id)
            # index of the last epoch, used when resuming a training job
            self.scheduler.last_epoch = self.current_epoch - 1

        '''print summary of the model when training'''
        if self.config.mode == 'train':
            batch_size = self.config.batch_size
            agent_num = self.config.agent_number
            channel_num = 3
            H, W = self.config.FOV + 2, self.config.FOV + 2
            dummy_GSO = torch.ones(size=(batch_size, agent_num, agent_num),
                                   device=self.config.device,
                                   dtype=torch.float)
            self.model.set_gso(dummy_GSO)
            summary(self.model,
                    torch.zeros((batch_size, agent_num, channel_num, H, W),
                                device=self.config.device),
                    batch_size=batch_size,
                    show_input=True,
                    max_depth=3,
                    show_hierarchical=True,
                    print_summary=True,
                    show_parent_layers=True)

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
                        best: bool = False,
                        latest: bool = False,
                        epoch: int = 0,
                        ) -> None:
        """
        Checkpoint saver
        Default behaviour: save as epoch 0
        :param epoch: current epoch being saved
        :param best: flag to indicate whether current checkpoint's metric is the best so far
        :param latest: flag to indicate the checkpoint is the latest one trained
        """
        if best:
            file_name = 'model_best.pth.tar'    # best checkpoint
        elif latest:
            file_name = 'checkpoint.pth.tar'    # latest checkpoint -> unnamed
        else:
            file_name = f'checkpoint_{epoch:03d}.pth.tar'   # name checkpoint

        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        # save the state
        torch.save(obj=state,
                   f=os.path.join(self.config.checkpoint_dir, file_name),
                   pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self,
                        best: bool = False,
                        latest: bool = False,
                        epoch: int = 0,
                        ) -> None:
        """
        Checkpoint loader
        Default behaviour: load epoch 0
        :param epoch: current epoch being loaded
        :param best: flag to indicate whether loading best checkpoint or not
        :param latest: flag to indicate the loaded checkpoint is the latest one trained
        """
        if best:
            file_name = 'model_best.pth.tar'
        elif latest:
            file_name = 'checkpoint.pth.tar'
        else:
            file_name = f'checkpoint_{epoch:03d}.pth.tar'

        try:
            self.logger.info(f'Loading checkpoint {file_name}')
            # load checkpoint, moving tensors onto selected device (cuda or cpu)
            checkpoint = torch.load(f=os.path.join(self.config.checkpoint_dir, file_name),
                                    map_location=torch.device(f'{self.config.device}'))

            # load back parameters
            self.current_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.logger.info(f'Checkpoint loaded successfully from {self.config.checkpoint_dir}'
                             f'at epoch {checkpoint["epoch"]}\n')
        # no file found
        except OSError:
            self.logger.warning(f'Desired checkpoint {file_name} does not exist in {self.config.checkpoint_dir}'
                                f'Skipping')

    def run(self) -> None:
        """
        The main operator
        """
        try:
            # training mode
            if self.config.mode == 'train':
                self.train()

            # testing mode
            elif self.config.mode == 'test':
                start_time = timeit.default_timer()
                self.test()
                self.time_record = timeit.default_timer() - start_time

            # validation only mode
            else:
                # list of all ckp names
                ckp_list = [os.path.join(self.config.checkpoint_dir, filename)
                            for filename in os.listdir(self.config.checkpoint_dir)
                            if filename != 'checkpoint.pth.tar'
                            and filename != 'model_best.pth.tar']
                # for each saved model
                for ckp_path in ckp_list:
                    performance = self.validate(checkpoint_path=ckp_path)
                    self.logger.info(f'Validation {performance}')

                    # if performance was the best
                    if performance > self.best_performance:
                        # save performance value and best checkpoint
                        self.best_performance = performance.copy()
                        shutil.copyfile(ckp_path,
                                        os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))

        # interrupting training or testing by keyboard
        except KeyboardInterrupt:
            self.logger.info('Entered CTRL+C. Finalizing')

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
            # save the latest model checkpoint for eventual validation or restarting
            self.save_checkpoint(latest=True)

            # validate only every n epochs
            if epoch % self.config.validate_every == 0:
                # save the checkpoint corresponding to validation
                self.save_checkpoint(epoch=epoch)

                if not self.config.skip_valid:
                    # do validation
                    performance = self.validate(checkpoint_path=os.path.join(self.config.checkpoint_dir,
                                                                             'checkpoint.pth.tar'))
                    self.logger.info(f'Validation {performance}')

                    # if performance was the best
                    if performance > self.best_performance:
                        # save performance value and best checkpoint
                        self.best_performance = performance.copy()
                        self.save_checkpoint(best=True)

            # scheduler step
            self.scheduler.step()

    def validate(self,
                 checkpoint_path: str
                 ) -> metrics.Performance:
        """
        Validate model saved in the given checkpoint
        :param checkpoint_path: path to the checkpoint to load model from
        :return: mean performance recorder during the validation simulation
        """
        self.logger.info(f'Start validation: '
                         f'Model loaded from {os.path.basename(checkpoint_path)}')
        data_loader = self.data_loader.valid_loader     # get valid loader
        # return mean performance of the simulation
        return self.simulate_agent_exec(data_loader=data_loader,
                                        checkpoint_path=checkpoint_path)

    def test(self) -> None:
        """
        Main testing loop
        Uses model save in 'best_checkpoint'
        """
        self.logger.info('Start testing')
        data_loader = self.data_loader.test_loader  # get test loader
        # simulate
        mean_performance = self.simulate_agent_exec(data_loader=data_loader,
                                                    checkpoint_path=os.path.join(self.config.checkpoint_dir,
                                                                                 'model_best.pth.tar'))
        # set mean performance for printing
        self.best_performance = mean_performance

    def finalize(self) -> None:
        """
        Concluding all operations and printing results
        """
        if self.config.mode == 'test':
            self.logger.info('################## End of testing ##################')
            self.logger.info(f'Best {self.best_performance}')
            self.logger.info(f'Computation time:\t{self.time_record} ')

        # train mode
        elif self.config.mode == 'train':
            self.logger.info('################## End of training ##################')
            if not self.config.skip_valid:
                self.logger.info(f'Best Validation {self.best_performance}')
            else:
                self.logger.info(f'Validation was not performed as selected')

        # validation mode
        else:
            self.logger.info('################## End of validation ##################')
            self.logger.info(f'Best Validation {self.best_performance}')

    def train_one_epoch(self) -> None:
        """
        One epoch of training
        """

        # set the model to be in training mode
        self.model.train()

        # loss accumulator
        running_loss = 0.0
        logged_batch = 0

        # loop over various batches of training data
        for batch_idx, (batch_input, batch_GSO, batch_target, _) \
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
            batch_target = batch_target.reshape(B * N, -1)

            # init model and loss
            self.model.set_gso(batch_GSO)

            # set zero grad for the optimizer
            self.optimizer.zero_grad()

            # AMP optimization
            with torch.cuda.amp.autocast(enabled=self.amp):
                # get model prediction, B*N x 5
                predict = self.model(batch_input)
                # compute loss
                # torch.max returns both values and indices
                # torch.max axis = 1 -> find the index of the chosen action for each agent
                loss = self.loss_f(predict, torch.max(batch_target, 1)[1])  # [1] to unpack indices

            # update gradient with backward pass using AMP scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            logged_batch += 1
            # log progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(f'Epoch {self.current_epoch}:'
                                 f'[{batch_idx * self.config.batch_size}/{len(self.data_loader.train_loader.dataset)}'
                                 f'({100 * batch_idx / len(self.data_loader.train_loader):.0f}%)] '
                                 f'- Learning Rate: {self.scheduler.get_last_lr()}\t'
                                 f'Loss: {running_loss / logged_batch:.6f}')
                running_loss = 0.0
                logged_batch = 0

        # always last batch logged
        self.logger.info(f'Epoch {self.current_epoch}:'
                         f'[{len(self.data_loader.train_loader.dataset)}/{len(self.data_loader.train_loader.dataset)}'
                         f'({100.:.0f}%)]\t'
                         f'Loss: {running_loss / (logged_batch if logged_batch > 0 else 1):.6f}')

    def sim_agent_exec_single(self,
                              data_loader: data.DataLoader,
                              checkpoint_path: str,
                              ) -> metrics.Performance:
        """
        Simulate all MAPD problem in the data loader using trained model for solving it
        All the simulations are done in a single process, sequentially
        :param data_loader: pytorch Dataloader, either testing or valid DataLoader
        :param checkpoint_path: path to the checkpoint to load model from
        :return mean performance recorder during the validation simulation
        """
        self.logger.info('Starting single process MAPD simulations')

        # simulation handling class, performance recorder and dataset aggregation
        simulator = sim.MultiAgentSimulator(config=self.config,
                                            device=self.config.device)
        recorder = metrics.PerformanceRecorder(simulator=simulator)
        aggregator = agg.Aggregator(config=self.config)

        performance_list: List[metrics.Performance] = []

        # load the model parameters
        checkpoint = torch.load(f=checkpoint_path,
                                map_location=torch.device(f'{self.config.device}'))

        model = gatp.GaTpNet(config=self.config).to(self.config.device)
        model.load_state_dict(checkpoint['state_dict'])

        with torch.no_grad():
            # loop over all cases in the test/valid data loader
            for case_idx, (obstacle_map, start_pos_list, task_list, makespan, service_time, basename) \
                    in enumerate(data_loader):
                # simulate the MAPD execution
                # batch size = 1 -> unpack all tensors
                simulator.simulate(obstacle_map=obstacle_map[0],
                                   start_pos_list=start_pos_list[0],
                                   task_list=task_list[0],
                                   model=model,
                                   target_makespan=makespan.item())

                # collect metrics
                performance = recorder.evaluate_performance(target_makespan=makespan.item())
                performance_list.append(performance)
                metrics.print_performance(performance=performance,
                                          mode=self.config.mode,
                                          logger=self.logger,
                                          case_idx=case_idx,
                                          max_size=len(data_loader),
                                          print_valid_every=self.config.print_valid_every)

                # collect challenging configurations from the current validation case
                aggregator.collect_cases(simulator=simulator,
                                         valid_basename=basename[0],  # list of 1, due to the batch size
                                         epoch=self.current_epoch)

        # extend training dataset with new solved cases
        aggregator.extend_dataset(dataset=self.data_loader.train_dataset
                                          if self.config.mode == 'train'
                                          else None)

        # return average performances
        return metrics.get_avg_performance(performance_list=performance_list)

    def sim_agent_exec_multi(self,
                             data_loader: data.DataLoader,
                             checkpoint_path: str,
                             ) -> metrics.Performance:
        """
        Simulate all MAPD problem in the data loader using trained model for solving it
        Multiprocessing simulation
        :param data_loader: pytorch Dataloader, either testing or valid DataLoader
        :param checkpoint_path: path to the checkpoint to load model from
        :return mean performance recorder during the validation simulation
        """
        self.logger.info('Starting multi process MAPD simulations')

        # useful variables
        data_size = len(data_loader)

        # load data from data loader
        with torch.no_grad():
            # set up queues
            performance_queue = Queue(ctx=mp.get_context('spawn'),
                                      maxsize=data_size + 1)
            data_queue = Queue(ctx=mp.get_context('spawn'),
                               maxsize=data_size + 1)
            aggregation_queue = Queue(ctx=mp.get_context('spawn'))

            # fill data queue
            for i, data_ in enumerate(data_loader):
                data_queue.put((i, data_), block=True)
            # collect args for multiprocessing
            # config, data queue, performance queue, aggregation_queue, checkpoint_path, current_epoch
            args = (self.config,
                    data_queue,
                    performance_queue,
                    aggregation_queue,
                    checkpoint_path,
                    self.current_epoch)

            # spawn and run processes, wait all to finish
            mp.spawn(fn=sim_worker,
                     args=args,
                     nprocs=self.config.sim_num_process,
                     join=True)

            # get performance list
            performance_queue.put(STOP_SENTINEL, block=True)  # termination sentinel
            performance_list = [p for p in iter(performance_queue.get, STOP_SENTINEL)]

            # release data queue
            data_queue.close()
            data_queue.join_thread()
            # release performance queue
            performance_queue.close()
            performance_queue.join_thread()

            # get list of conflicting cases joined amongst all processes
            aggregator = agg.Aggregator(config=self.config)
            aggregation_queue.put(STOP_SENTINEL, block=True)  # termination sentinel
            aggregator.cases_list = [case for case_list in iter(aggregation_queue.get, STOP_SENTINEL)
                                     for case in case_list]
            aggregation_queue.close()
            aggregation_queue.join_thread()

            # extend training dataset
            aggregator.extend_dataset(dataset=self.data_loader.train_dataset
                                              if self.config.mode == 'train'
                                              else None)

        # return average performances
        return metrics.get_avg_performance(performance_list=performance_list)


def sim_worker(process_id: int,
               config: EasyDict,
               data_queue: Queue,
               performance_queue: Queue,
               aggregation_queue: Queue,
               checkpoint_path: str,
               current_epoch: int
               ) -> None:
    """
    Class for multiprocessing simulation
    Simulate MAPD execution and record performances over input data taken from iterating over a dataloader
    :param process_id: id of the process that executes the function, automatically passed by mp spawn
    :param config: Namespace of configurations
    :param data_queue: contains shared input data to run simulation over
    :param performance_queue: shared queue to put simulation results
    :param aggregation_queue: shared queue to put conflicting cases for dataset augmentation
    :param checkpoint_path: path to the checkpoint to load model from
    :param current_epoch: id of the current training epoch, used for naming dagger cases
    """
    # simulation handling class, performance recorder and dataset aggregation
    simulator = sim.MultiAgentSimulator(config=config,
                                        device=config.device)
    recorder = metrics.PerformanceRecorder(simulator=simulator)
    aggregator = agg.Aggregator(config=config)

    # data size
    data_size = data_queue.qsize()

    print(f'Process {process_id} successfully initialized')

    with torch.no_grad():
        while not data_queue.empty():
            # unpack tensors from input data queue
            case_idx, (obstacle_map, start_pos_list, task_list,
                       makespan, service_time, basename) = data_queue.get(block=True)

            # load the model parameters
            checkpoint = torch.load(f=checkpoint_path,
                                    map_location=torch.device(f'{config.device}'))

            model = gatp.GaTpNet(config=config).to(config.device)
            model.load_state_dict(checkpoint['state_dict'])

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
            if case_idx % config.print_valid_every == 0:
                print(f'Validation at {100 * case_idx / data_size:.0f}%')

            # collect challenging configurations from the current validation case
            aggregator.collect_cases(simulator=simulator,
                                     valid_basename=basename,
                                     epoch=current_epoch)

            # add metrics
            performance_queue.put(performance, block=True)

    # add collect cases in the process for dataset extension
    aggregation_queue.put(aggregator.cases_list.copy(), block=True)
