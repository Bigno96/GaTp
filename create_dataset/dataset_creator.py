"""
Main function for creating datasets
If a dataset with a different expert solution already exist, new expert solutions are added to the existing dataset
Folder structure:
datasets
    |-> map_type
        |-> map_dimension
            |-> obstacle_density
                |-> agents and tasks number
                    |-> task and starting position creation method
                        |-> training
                        |-> validation
                        |-> testing

All parameters are controlled in GaTp/yaml_configs/Dataset_creation.yaml

Inside each training, validation and testing folders there are:
- image of the map + starting positions
- file with {'map', 'non_task_endpoint_list', 'task_list'}
- file with the representation of the agents schedule obtained by the expert (different experts, different files)
"""

import os

from utils.config import get_config_from_yaml
from utils.file_utils import create_dirs
from create_dataset.data_pool_creator import create_data_pool


def create_dataset():
    """
    Create dataset consisting of maps, scenarios and experts solutions
    """
    # get config from yaml file
    config = get_config_from_yaml("Dataset_creation")

    # create folder for the dataset
    dataset_dir = os.path.join(config.data_root,
                               'random_grid',
                               f'{config.map_shape[0]}x{config.map_shape[1]}map',
                               f'{config.map_density}density',
                               f'{config.agent_number}agents_{config.task_number}tasks',
                               f'{config.start_position_mode}_start+{config.task_creation_mode}_task')

    create_dirs([dataset_dir])

    # create maps with scenarios
    create_data_pool(config=config, dataset_dir=dataset_dir)


if __name__ == '__main__':
    __spec__ = None
    create_dataset()
