"""
PyTorch Custom Dataset implementation

Functions and methods to create Dataset instance over all solved MAPD instances
"""

import os
import logging
import torch

from torch.utils.data import Dataset
from data_loading.transform_data import DataTransformer
from multiprocessing.pool import Pool
from itertools import repeat


class GaTpDataset(Dataset):

    def __init__(self, config, mode):
        self.config = config
        self.logger = logging.getLogger("Dataset")

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

        self.data_transform = DataTransformer(config=config, data_path=self.data_path)      # transforming input data
        self.data_cache = {}        # dictionary for data caching

        self.logger.info('Start loading data')

        # train mode
        if mode == "train":

            # list of case files (only names, not full path)
            self.basename_list = self.load_basename_list(mode=mode)

            # load data into cache
            num_loaded = 0
            total_cases = len(self.basename_list)
            for basename in self.basename_list:
                # nn_data = input tensor, GSO, target
                nn_data = self.data_transform.get_nn_data(basename=basename, mode=mode)     # get data
                self.data_cache[basename] = nn_data     # dump into cache
                num_loaded += 1
                print(f'Loaded train data {num_loaded}/{total_cases}')

            # get a mapping of basename and their makespan
            # when index is given -> return associated basename and timestep
            self.basename_switch = BasenameSwitch(basename_list=self.basename_list,
                                                  data_cache=self.data_cache)
            # data size
            self.data_size = self.basename_switch.data_size

        else:
            pass

    def __getitem__(self, index):
        """
        :param index: int
        :return: step_input_tensor, step_GSO, step_target
                 step_input_tensor -> FloatTensor,
                                      shape = (makespan, agent_num, channel_num, FOV+2*border, FOV+2*border)
                 step_GSO -> FloatTensor,
                             shape = (makespan, agent_num, agent_num)
                 step_target -> FloatTensor,
                                shape = (makespan, num_agent, 5)
        """
        # obtain corresponding case name and timestep of its solution
        basename, timestep = self.basename_switch.get_item(index)
        # get nn_data
        input_tensor, GSO, target = self.data_cache[basename]

        # slice 1 timestep
        step_input_tensor = input_tensor[timestep].float()      # already torch tensor, cast to float for good measure
        step_GSO = torch.from_numpy(GSO[timestep]).float()
        step_target = torch.from_numpy(target[timestep]).float()

        return step_input_tensor, step_GSO, step_target

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
            # look up for makespan in the data cache
            _, GSO, _ = data_cache[basename]
            makespan = GSO.shape[0]        # first dim is the makespan

            # basename is associated with a range
            self.switch[range(cum_makespan, cum_makespan+makespan)] = basename

            cum_makespan += makespan        # sum up

        self.data_size = cum_makespan

    def get_item(self, idx):
        """
        Return basename and timestep (of the corresponding solution) selected by index
        :param idx: int, index of __getitem__
        :return: str, int
                 basename, timestep
        """
        for key_range, value in self.switch.items():
            if idx in key_range:
                # return basename, timestep
                return value, idx % key_range[0]


'''class GaTpDataset(Dataset):
    """
    Custom Dataset for loading MAPD data to feed into GaTp
    """
    def __init__(self, data_dir, mode, expert_type):
        """
        :param data_dir: path of the directory containing data
        :param mode: dataset type -> train, valid or test
        :param expert_type: type of expert to load the data -> tp
        """
        # decide which folder to use
        if mode.upper() == 'TRAIN':
            self.data_dir = os.path.join(data_dir, 'train')
        elif mode.upper() == 'TEST':
            self.data_dir = os.path.join(data_dir, 'test')
        elif mode.upper() == 'VALID':
            self.data_dir = os.path.join(data_dir, 'valid')
        else:
            raise ValueError('Invalid Dataset mode')

        # select expert
        if expert_type.upper() == 'TP':
            self.exp_sol_extension = 'tp_sol'
        else:
            raise ValueError('Invalid Expert type')

        # get list of base names
        self.name_list = [filename
                          for filename in os.listdir(self.data_dir)
                          if not filename.endswith('.png')
                          and not filename.endswith('sol')]

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.name_list)

    def __getitem__(self, index):
        """
        :param index: int
        :return: image of the environment -> matrix
                 environment description -> as a Namespace
                 expert solution -> as a Namespace
        """
        base_name = self.name_list[index]

        # load image
        img_path = os.path.join(self.data_dir, f'{base_name}.png')
        image = read_image(img_path).T

        # load environment
        env_path = os.path.join(self.data_dir, base_name)
        with open(env_path, 'rb') as _data:
            environment = pickle.load(_data)

        # load expert solution
        sol_path = os.path.join(self.data_dir, f'{base_name}_{self.exp_sol_extension}')
        with open(sol_path, 'rb') as _data:
            expert_sol = pickle.load(_data)

        return image, environment, expert_sol'''
