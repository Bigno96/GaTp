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
                basename -> name of the dataset environment file

 valid/testing item -> (start_pos_list, task_list, makespan, service_time, basename)
                    start_pos_list -> starting positions of the agents
                                      shape = (agent_num, 2)
                    task_list -> task list, each task identified by 2 positions (pickup, delivery)
                                 shape = (task_num, 2, 2)
                    makespan -> length of the solution found by the expert
                    service_time -> average timesteps required to complete a task by the expert
                    basename -> name of the dataset environment file
"""

import os
import logging
import torch
import pickle
from tqdm import tqdm
from p_tqdm import p_map

from torch.utils.data import Dataset
from utils.transform_data import DataTransformer
from utils.file_utils import load_basename_list


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

    def __init__(self, config, mode):
        """
        :param config: Namespace of dataset configurations
        :param mode: str, options: ['test', 'train', 'valid']
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
        self.data_transform = DataTransformer(config=config, data_path=self.data_path, mode=self.mode)

        self.logger.info(f'Start loading {self.mode} data')
        # list of case files (only names, not full path)
        self.basename_list = load_basename_list(data_path=self.data_path, mode=self.mode)

        # if input data have to be generated at data loading
        if self.config.transform_runtime_data:
            # get processed data
            result = p_map(self.data_transform.get_data, self.basename_list)
        # read input data from datasets file
        else:
            result = []
            for basename in tqdm(self.basename_list):
                # get 'data' file path
                data_path = os.path.join(self.data_path, self.mode, f'{basename}_data')
                # extract with pickle
                with open(data_path, 'rb') as f:
                    result.append(pickle.load(f))

        # dictionary for data caching
        self.data_cache = dict(zip(self.basename_list, result))

        # get a mapping of basename and their makespan
        # when index is given -> return associated basename and timestep
        self.basename_switch = BasenameSwitch(basename_list=self.basename_list,
                                              data_cache=self.data_cache,
                                              mode=self.mode)
        # data size
        self.data_size = self.basename_switch.data_size

    def __getitem__(self, index):
        """
        :param index: int
        :return: training -> (step_input_tensor, step_GSO, step_target, basename)
                        step_input_tensor -> IntTensor,
                                             shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                        step_GSO -> FloatTensor,
                                    shape = (agent_num, agent_num)
                        step_target -> FloatTensor,
                                       shape = (num_agent, 5)
                        basename -> str

                 valid/testing -> (start_pos_list, task_list, makespan, service_time, basename)
                        start_pos_list -> IntTensor,
                                          shape = (agent_num, 2)
                        task_list -> IntTensor,
                                     shape = (task_num, 2, 2)
                        makespan -> int
                        service_time -> float
                        basename -> str
        """
        # obtain corresponding case name and timestep of its solution
        basename, timestep = self.basename_switch.get_item(index)

        # get train data
        if self.mode == 'train':
            step_input_tensor, step_GSO, step_target = self.get_train_data(basename=basename, timestep=timestep)
            item = (step_input_tensor, step_GSO, step_target, basename)

        # get test/valid data
        else:
            start_pos_list, task_list, makespan, service_time = self.get_test_data(basename=basename)
            item = (start_pos_list, task_list, makespan, service_time, basename)

        return item

    def __len__(self):
        return self.data_size

    def get_train_data(self, basename, timestep):
        """
        Retrieve training data from data cache
        :param basename: str, case file name
        :param timestep: int, timestep of the solution associated to the case
        :return: step_input_tensor, step_GSO, step_target
                 step_input_tensor -> IntTensor,
                                      shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                 step_GSO -> FloatTensor,
                             shape = (agent_num, agent_num)
                 step_target -> FloatTensor,
                                shape = (num_agent, 5)
        """
        # get train_data
        input_tensor, GSO, target = self.data_cache[basename]

        # slice 1 timestep
        step_input_tensor = input_tensor[timestep].int()  # already torch tensor, cast to int for good measure
        step_GSO = torch.from_numpy(GSO[timestep]).float()
        step_target = torch.from_numpy(target[timestep]).long()

        return step_input_tensor, step_GSO, step_target

    def get_test_data(self, basename):
        """
        Retrieve testing data from data cache
        :param basename: str, case file name
        :return: start_pos_list, task_list, makespan, service_time
                 start_pos_list -> IntTensor,
                                   shape = (agent_num, 2)
                 task_list -> IntTensor,
                              shape = (task_num, 2, 2)
                 makespan -> int
                 service_time -> float
        """
        # get test_data
        start_pos_list, task_list, makespan, service_time = self.data_cache[basename]

        # convert to tensor
        start_pos_list = torch.from_numpy(start_pos_list).int()
        task_list = torch.from_numpy(task_list).int()

        return start_pos_list, task_list, makespan, service_time


class BasenameSwitch:
    """
    During training, every time step of a solution makes for a different input
    Therefore, every basename is repeated n times,
    where n is the makespan of the expert solution associated to the case

    For validation and testing, a basename is used only once

    This class convert index of __getitem__ to the corresponding basename and timestep
    """

    def __init__(self, basename_list, data_cache, mode):
        """
        :param basename_list: list of str, all case file's names
        :param data_cache: dictionary, {basename : nn_data}
                                       nn_data = input tensor, GSO, target
        :param mode: str, options: ['test', 'train', 'valid']
        """
        self.switch = {}        # dictionary with range as keys

        # training mode
        if mode == 'train':
            # for each basename
            cum_makespan = 0        # sum of length up to now
            for basename in basename_list:
                # look up for makespan in the data cache
                _, GSO, _ = data_cache[basename]
                makespan = GSO.shape[0]        # first dim is the makespan

                # basename is associated with a range
                self.switch[range(cum_makespan, cum_makespan+makespan)] = basename

                cum_makespan += makespan        # sum up

            self.data_size = cum_makespan

        # testing/valid mode
        else:
            # basename is associated with a range of length 1
            for i, basename in enumerate(basename_list):
                self.switch[range(i, i+1)] = basename

            self.data_size = len(basename_list)  # each basename used once

    def get_item(self, idx):
        """
        Return basename and timestep (of the corresponding solution) selected by index
        :param idx: int, index of __getitem__
        :return: str, int
                 basename, timestep
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
