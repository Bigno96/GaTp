"""
PyTorch Custom Dataset implementation

Functions and methods to create Dataset instance over all solved MAPD instances
"""

from torch.utils.data import Dataset
import pickle
import os
from torchvision.io import read_image


class GaTpDataset(Dataset):
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

        return image, environment, expert_sol
