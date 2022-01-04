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
- file with environment data:
        {'name', 'map', 'start_pos_list, 'parking_spot_list', 'task_list'}
- file with the representation of the agents schedule obtained by the expert (different experts, different files):


## More information at GaTp/create_dataset/environment_creator.py and GaTp/create_dataset/run_expert.py ##
"""

import logging
import os

from create_dataset.environment_creator import create_environment
from create_dataset.run_expert import run_expert
from utils.config import get_config_from_yaml


def create_dataset():
    """
    Create dataset consisting of maps, scenarios and experts solutions
    """
    # get config from yaml file
    config = get_config_from_yaml('Dataset_creation')

    # create folder for the dataset
    dataset_dir = os.path.join(config.data_root,
                               'random_grid',
                               f'{config.map_shape[0]}x{config.map_shape[1]}map',
                               f'{config.map_density}density',
                               f'{config.agent_number}agents_{config.task_number}tasks',
                               f'{config.start_position_mode}_start+{config.task_creation_mode}_task')

    try:
        # create maps with scenarios
        create_environment(config=config, dataset_dir=dataset_dir)
        # run expert over those environments
        run_expert(config=config, dataset_dir=dataset_dir)

    # invalid configuration parameters passed
    except ValueError as err:
        logging.getLogger().warning(err)
        exit(-1)


if __name__ == '__main__':
    __spec__ = None
    create_dataset()
