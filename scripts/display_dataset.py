import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


def display(dataset_path):

    train_dataset = VisualDataset(data_dir=dataset_path, mode='train', expert_type='tp')
    valid_dataset = VisualDataset(data_dir=dataset_path, mode='valid', expert_type='tp')
    test_dataset = VisualDataset(data_dir=dataset_path, mode='test', expert_type='tp')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print('Showing train dataset')
    iterate_data(dataloader=train_dataloader)

    print('Showing validation dataset')
    iterate_data(dataloader=valid_dataloader)

    print('Showing test dataset')
    iterate_data(dataloader=test_dataloader)


def iterate_data(dataloader):

    for image, environment, expert_sol, nn_data in dataloader:

        plt.imshow(image.squeeze())
        plt.show()

        # batch size = 1 -> unpack with [0]
        print(f'name: {environment["name"][0]}')
        print(f'expert_name: {expert_sol["name"][0]}')

        print(f'\nmap: \n {environment["map"][0].numpy()}')
        print(f'\nstart_pos_list: {convert_to_list(environment["start_pos_list"])}')
        print(f'parking_spot_list: {convert_to_list(environment["parking_spot_list"])}')
        print(f'task_list: {convert_task(environment["task_list"])}')

        print('\nagent_schedule:')
        for id_ag, sched in expert_sol["agent_schedule"].items():
            print(f'{id_ag}: {convert_to_list(sched)}')

        print('\ngoal_schedule:')
        for id_ag, sched in expert_sol["goal_schedule"].items():
            print(f'{id_ag}: {convert_to_list(sched)}')

        print(f'\nmakespan: {expert_sol["makespan"][0]}')
        print(f'service_time: {expert_sol["service_time"][0]}')
        print(f'runtime_per_timestep: {expert_sol["runtime_per_timestep"][0]}')
        print(f'collisions: {expert_sol["collisions"][0]}')

        input('\nPress Enter to continue\n')


def convert_to_list(tensor_list):
    ret = []
    for e in tensor_list:
        L = [el.tolist()[0] for el in e]
        ret.append(L)
    return ret


def convert_task(tensor_list):
    ret = []
    for e in tensor_list:
        task = []
        for el in e:
            L = [elem.tolist()[0] for elem in el]
            task.append(L)
        ret.append(task)
    return ret


class VisualDataset(Dataset):
    """
    Custom Dataset for Visualizing Datasets
    """
    def __init__(self, data_dir, mode: str, expert_type):
        """
        :param data_dir: path of the directory containing data
        :param mode: dataset type -> train, valid or test
        :param expert_type: type of expert to load the data -> tp
        """
        assert mode in ['train', 'test', 'valid']
        # decide which folder to use
        self.data_dir = os.path.join(data_dir, mode)

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

        # load nn data
        nn_path = os.path.join(self.data_dir, f'{base_name}_data')
        with open(nn_path, 'rb') as _data:
            nn_data = pickle.load(_data)

        return image, environment, expert_sol, nn_data


if __name__ == '__main__':
    __spec__ = None
    np.set_printoptions(threshold=sys.maxsize)

    data_path = 'D:/Uni/TESI/GaTp/datasets/random_grid/20x20map/0.1density/20agents_500tasks_0.0split_+1_every1/random_start+avoid_non_task_rep_task'
    display(dataset_path=data_path)
