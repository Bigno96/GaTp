"""
PyTorch Custom Dataset implementation

Functions and methods to create torch Dataset over all solved MAPD instances

DataTransformer is used to modify saved data to build tensor to feed to the model

training item -> (step_input_tensor, step_GSO, step_target, basename)
                step_input_tensor -> input tensor with the different channels to feed to the ML model
                                     shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                step_GSO -> Graph Shift Operator, implemented as Normalized Adjacency Matrix
                            shape = (agent_num, agent_num)
                step_target -> matrix form of agent's action schedule, 5 possible actions
                               shape = (num_agent, 5)
                basename -> str, case file name

 valid/testing item -> (obstacle_map, start_pos_list, task_list, makespan, service_time, basename)
                    obstacle_map -> map of obstacles,
                                    shape = (H, W)
                    start_pos_list -> starting positions of the agents
                                      shape = (agent_num, 2)
                    task_list -> task list, each task identified by 2 positions (pickup, delivery)
                                 shape = (task_num, 2, 2)
                    makespan -> length of the solution found by the expert
                    service_time -> average timesteps required to complete a task by the expert
                    basename -> str, case file name
"""

import os
import logging
import torch
import pickle

import numpy as np
import utils.transform_data as tf_data
import utils.file_utils as f_utils

from p_tqdm import p_map
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict
from typing import List, Tuple, Dict


class GaTpDataset(Dataset):
    """
    Custom dataset for GaTp model
    init -> load dataset environments and expert solutions from pickle files
            transform data in order to feed them to the ML model
            save transformed data into a data cache
    getitem -> retrieve correct data to pass to the model, based on current mode (train, test, valid)
               use 'index' parameter to collect proper case base name
               basename is used to access data cache
    """

    def __init__(self,
                 config: EasyDict,
                 mode: str):
        """
        :param config: Namespace of dataset configurations
        :param mode: 'test', 'train' or 'valid'
        """
        self.config = config
        self.logger = logging.getLogger("Dataset")

        # check mode type
        assert mode in ['train', 'valid', 'test']
        self.mode = mode

        # path to datasets root folder
        self.data_root = self.config.data_root
        # get data folder using arguments specification
        self.data_path = os.path.join(self.data_root,
                                      f'{config.map_type}',
                                      f'{config.map_shape[0]}x{config.map_shape[1]}map',
                                      f'{config.map_density}density',
                                      f'{config.agent_number}agents_{config.task_number}tasks_'
                                      f'{config.imm_task_split}split_'
                                      f'+{config.new_task_per_timestep}_every{config.step_between_insertion}',
                                      f'{config.start_position_mode}_start+{config.task_creation_mode}_task'
                                      )

        # class for transforming input data
        self.data_transform = tf_data.DataTransformer(config=config,
                                                      data_path=self.data_path,
                                                      mode=self.mode)

        self.logger.info(f'Start loading {self.mode} data')
        # list of case files (only names, not full path)
        self.basename_list = f_utils.load_basename_list(data_path=self.data_path,
                                                        mode=self.mode)

        # process data at loading time
        if self.config.transform_runtime_data:
            result = p_map(self.data_transform.get_data, self.basename_list)
        # read processed data from datasets file
        else:
            result = p_map(self.__load_cache_data, self.basename_list)

        # dictionary for data caching
        self.data_cache = dict(zip(self.basename_list, result))

        # get a mapping of basename and their makespan
        # when index is given -> return associated basename and timestep
        self.basename_switch = BasenameSwitch(basename_list=self.basename_list,
                                              data_cache=self.data_cache,
                                              mode=self.mode)

        # item getter
        if self.mode == 'train':
            self.get_data = self.get_train_data
        else:
            self.get_data = self.get_test_data

    def __load_cache_data(self,
                          basename: str
                          ) -> Tuple[np.array or int or float, ...]:
        """
        Used to read already transformed data from file path
        :param basename: case file name
        :return: train or testing data, loaded from 'data' file
        """
        # get 'data' file path
        data_path = os.path.join(self.data_path, self.mode, f'{basename}_data')
        # extract with pickle
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self,
                    index: int
                    ) -> Tuple[torch.Tensor, ...]:
        """
        :param index: int
        :return: training -> (step_input_tensor, step_GSO, step_target, basename)
                        step_input_tensor -> FloatTensor,
                                             shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                        step_GSO -> FloatTensor,
                                    shape = (agent_num, agent_num)
                        step_target -> FloatTensor,
                                       shape = (num_agent, 5)
                        basename -> str, case file name

                 valid/testing -> (obstacle_map, start_pos_list, task_list, makespan, service_time, basename)
                        obstacle_map -> FloatTensor,
                                        shape = (H, W)
                        start_pos_list -> FloatTensor,
                                          shape = (agent_num, 2)
                        task_list -> FloatTensor,
                                     shape = (task_num, 2, 2)
                        makespan -> int
                        service_time -> float
                        basename -> str, case file name
        """
        # obtain corresponding case name and timestep of its solution
        basename, timestep = self.basename_switch.get_item(index)
        # get data
        return self.get_data(basename=basename, timestep=timestep)

    def __len__(self) -> int:
        return self.basename_switch.data_size

    def get_train_data(self,
                       **kwargs: int or str
                       ) -> Tuple[torch.Tensor, ...]:
        """
        Retrieve training data from data cache
        :param **kwargs
                'basename': str, case file name
                'timestep': int, timestep of the solution associated to the case
        :return: step_input_tensor, step_GSO, step_target, basename
                 step_input_tensor -> FloatTensor,
                                      shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                 step_GSO -> FloatTensor,
                             shape = (agent_num, agent_num)
                 step_target -> FloatTensor,
                                shape = (num_agent, 5)
                 basename -> str, case file name
        """
        basename = kwargs.get('basename')
        timestep = kwargs.get('timestep')

        # get train_data
        input_tensor, GSO, target = self.data_cache[basename]

        # slice 1 timestep
        step_input_tensor = torch.from_numpy(input_tensor[timestep]).float()
        step_GSO = torch.from_numpy(GSO[timestep]).float()
        step_target = torch.from_numpy(target[timestep]).float()

        return step_input_tensor, step_GSO, step_target, basename

    def get_test_data(self,
                      **kwargs: int or str,
                      ) -> Tuple[torch.Tensor, ...]:
        """
        Retrieve testing data from data cache
        :param **kwargs
                'basename': str, case file name
        :return: obstacle_map, start_pos_list, task_list, makespan, service_time, basename
                 obstacle_map -> FloatTensor,
                                 shape = (H, W)
                 start_pos_list -> FloatTensor,
                                   shape = (agent_num, 2)
                 task_list -> FloatTensor,
                              shape = (task_num, 2, 2)
                 makespan -> int
                 service_time -> float
                 basename -> str, case file name
        """
        basename = kwargs.get('basename')

        # get test_data
        obstacle_map, start_pos_list, task_list, makespan, service_time = self.data_cache[basename]

        # convert to tensor
        obstacle_map = torch.from_numpy(obstacle_map).float()
        start_pos_list = torch.from_numpy(start_pos_list).float()
        task_list = torch.from_numpy(task_list).float()

        return obstacle_map, start_pos_list, task_list, makespan, service_time, basename

    def extend_dataset(self,
                       input_map_list: List[np.array],
                       expert_sol_list: List[Dict[str, Dict]],
                       basename_list: List[str]
                       ) -> None:
        """
        Used during dataset aggregation for extending training dataset with augmented entries
        :param input_map_list: obstacle map
        :param expert_sol_list: solutions of the problem given by an expert algorithm
        :param basename_list: case names
        """
        self.logger.info('Add new Training data to the dataset')
        for input_map, basename, expert_sol in zip(tqdm(input_map_list), basename_list, expert_sol_list):
            # get NN-ready data and add it to the cache
            self.data_cache[basename] = self.data_transform.get_online_train_data(input_map=input_map,
                                                                                  expert_sol=expert_sol)
            # index new entries
            self.basename_switch.extend_switch(basename=basename,
                                               data_cache=self.data_cache)


class BasenameSwitch:
    """
    During training, every time step of a solution makes for a different input
    Therefore, every basename is repeated n times,
    where n is the makespan of the expert solution associated to the case

    For validation and testing, a basename is used only once

    This class convert index of __getitem__ to the corresponding basename and timestep
    """

    def __init__(self,
                 basename_list: List[str],
                 data_cache: Dict[str, Tuple[np.array, ...]],
                 mode: str):
        """
        :param basename_list: all case file's names
        :param data_cache: dictionary, {basename : nn_data}
        :param mode: 'test', 'train' or 'valid'
        """
        self.switch = {}    # dictionary with range as keys
        self.mode = mode

        # training mode
        if self.mode == 'train':
            # for each basename
            cum_makespan = 0    # sum of length up to now
            for basename in basename_list:
                # look up for makespan in the data cache
                _, GSO, _ = data_cache[basename]
                makespan = GSO.shape[0]     # first dim is the makespan

                # basename is associated with a range
                self.switch[range(cum_makespan, cum_makespan+makespan)] = basename

                cum_makespan += makespan    # sum up

            self.data_size = cum_makespan

        # testing/valid mode
        else:
            # basename is associated with a range of length 1
            for i, basename in enumerate(basename_list):
                self.switch[range(i, i+1)] = basename

            self.data_size = len(basename_list)     # each basename used once

    def extend_switch(self,
                      basename: str,
                      data_cache: Dict[str, Tuple[np.array, ...]],
                      ) -> None:
        """
        Add entry to the training basename switch during online dataset aggregation
        :param basename: case name
        :param data_cache: dictionary, {basename : nn_data}
        """
        assert self.mode == 'train'
        # look up for makespan in the data cache
        _, GSO, _ = data_cache[basename]
        makespan = GSO.shape[0]  # first dim is the makespan

        # basename is associated with a range
        self.switch[range(self.data_size, self.data_size+makespan)] = basename

        self.data_size += makespan

    def get_item(self,
                 idx: int
                 ) -> Tuple[str, int]:
        """
        Return basename and timestep (of the corresponding solution) selected by index
        :param idx: index of __getitem__
        :return: basename, timestep
        """
        for key_range, value in self.switch.items():
            if idx in key_range:
                # avoid modulo by 0
                if key_range[0] == 0:
                    # return basename, timestep
                    return value, idx
                else:
                    # return basename, timestep
                    return value, idx % key_range[0]
