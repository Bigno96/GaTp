"""
PyTorch Custom Dataset implementation

Functions and methods to create torch Dataset over all solved MAPD instances

DataTransformer is used to modify saved data to build tensor to feed to the model
"""

import os
import logging
import torch
from p_tqdm import p_map

from torch.utils.data import Dataset
from data_loading.transform_data import DataTransformer


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

        # transforming input data
        self.data_transform = DataTransformer(config=config, data_path=self.data_path, mode=mode)

        # start loading message
        self.logger.info(f'Start loading {mode} data')

        # list of case files (only names, not full path)
        self.basename_list = self.load_basename_list(mode=mode)

        # get processed data
        result = p_map(self.data_transform.get_nn_data, self.basename_list)

        # dictionary for data caching
        self.data_cache = dict(zip(self.basename_list, result))

        # get a mapping of basename and their makespan
        # when index is given -> return associated basename and timestep
        self.basename_switch = BasenameSwitch(basename_list=self.basename_list,
                                              data_cache=self.data_cache)
        # data size
        self.data_size = self.basename_switch.data_size

        if mode == 'train':
            self.get_data = self.get_train_data
        else:   # valid or test
            self.get_data = self.get_test_data

    def __getitem__(self, index):
        """
        :param index: int
        :return: step_input_tensor, step_GSO, step_target
                 step_input_tensor -> FloatTensor,
                                      shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                 step_GSO -> FloatTensor,
                             shape = (agent_num, agent_num)
                 step_target -> FloatTensor,
                                shape = (num_agent, 5)
                 basename -> str
        """
        # obtain corresponding case name and timestep of its solution
        basename, timestep = self.basename_switch.get_item(index)
        # get nn_data
        input_tensor, GSO, target = self.get_data(basename=basename, timestep=timestep)

        return input_tensor, GSO, target, basename

    def __len__(self):
        return self.data_size

    def load_basename_list(self, mode):
        """
        Load a file basename list
        File basename -> 'mapID_caseID'
        :param mode: str, options: ['test', 'train', 'valid']
        :return: List of str
        """
        # test, train or valid folder
        data_dir = os.path.join(self.data_path, mode)
        # get all and only filenames
        (_, _, filenames) = next(os.walk(data_dir))

        # filter out 'sol' and 'png' -> only basename
        return [name
                for name in filenames
                if 'sol' not in name
                and 'png' not in name]

    def get_train_data(self, **kwargs):
        """
        Retrieve training data from data cache
        :param **kwargs ->
            basename: str, case file name
            timestep: int, timestep of the solution associated to the case
        :return: step_input_tensor, step_GSO, step_target
                 step_input_tensor -> FloatTensor,
                                      shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                 step_GSO -> FloatTensor,
                             shape = (agent_num, agent_num)
                 step_target -> FloatTensor,
                                shape = (num_agent, 5)
        """
        basename = kwargs.get('basename')
        timestep = kwargs.get('timestep')

        # get nn_data
        input_tensor, GSO, target = self.data_cache[basename]

        # slice 1 timestep
        step_input_tensor = input_tensor[timestep].float()  # already torch tensor, cast to float for good measure
        step_GSO = torch.from_numpy(GSO[timestep]).float()
        step_target = torch.from_numpy(target[timestep]).long()

        return step_input_tensor, step_GSO, step_target

    def get_test_data(self, **kwargs):
        """
        Retrieve testing data from data cache
        :param **kwargs ->
            basename: str, case file name
        :return: input_tensor, target
                 input_tensor -> FloatTensor,
                                 shape = (agent_num, channel_num, FOV+2*border, FOV+2*border)
                 GSO -> FloatTensor,
                        shape = (agent_num, agent_num)
                 target -> FloatTensor,
                           shape = (makespan, num_agent, 5)
        """
        basename = kwargs.get('basename')

        # get nn data
        input_tensor, GSO, target = self.data_cache[basename]

        # makespan x agent_num x 5
        target = torch.from_numpy(target).long()    # convert target into tensor

        return input_tensor.float(), GSO.float(), target


class BasenameSwitch:
    """
    Every time step of a solution makes for a different input
    Therefore, every basename is repeated n times,
    where n is the makespan of the expert solution associated to the case
    This class convert index of __getitem__ to the corresponding basename and timestep
    """

    def __init__(self, basename_list, data_cache):
        """
        :param basename_list: list of str, all case file's names
        :param data_cache: dictionary, {basename : nn_data}
                                       nn_data = input tensor, GSO, target
        """
        self.switch = {}        # dictionary with range as keys

        # for each basename
        cum_makespan = 0        # sum of length up to now
        for basename in basename_list:
            # look up for makespan in the data cache -> why use GSO?
            # because in valid and test, GSO is not used -> GSO.shape[0] = 1 for those
            # which is desirable, because in valid and test you return all goal/schedule at once, no time slicing
            _, GSO, _ = data_cache[basename]
            makespan = GSO.shape[0]        # first dim is the makespan

            # basename is associated with a range
            self.switch[range(cum_makespan, cum_makespan+makespan)] = basename

            cum_makespan += makespan        # sum up

        self.data_size = cum_makespan       # for valid/test, data size = cases number

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
