"""
Run to create a dataset
Command line argument:
    - random_grid: create random_grid dataset
Parameters for the environment and the dataset are specified in a yaml file
"""
import logging
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from create_dataset.map_creator import *
from create_dataset.scenario_creator import *
from utils.config import get_config_from_yaml
from utils.create_dirs import create_dirs


def save_image(dataset_dir, input_map, map_id, start_pos_list, sc_id):
    """
    Save image of the map (black and white) with agent starting positions (red)
    :param dataset_dir: path to the specific dataset directory
    :param input_map: np.ndarray, size: H*W, matrix of '0' and '1'
    :param map_id: int, identifier of map
    :param start_pos_list: list of tuples, (x,y) -> coordinates over the map
    :param sc_id: int, identifier of scenario
    """
    # copy to avoid modifications
    copy_map = input_map.copy()
    img_path = os.path.join(dataset_dir, f'map{map_id:03d}_case{sc_id:02d}.png')
    for pos in start_pos_list:
        copy_map[pos] = 2
    # white = 0, background | black = 1, obstacles | red = 2, agents starting positions
    plt.imsave(img_path, copy_map,
               cmap=ListedColormap(['white', 'black', 'red']))


def __random_grid_dataset():
    """
    Pipeline:

        Create Maps (see map_creator.py for more details):
            MAP_NUMBER different random grid maps for the same setting (size, density)

        Create Scenarios (see scenario_creator.py for more details):
            To have well-formed MAPD instances, create_task should be set with mode = 'no_start_repetition'
            This will guarantee a set of non-task endpoints >= num agents, being the agents' starting positions
            SCENARIO_NUMBER different scenarios for each map
            AGENT_NUMBER different starting positions for agents
            TASK_NUMBER series of tasks (pickup and delivery positions)
            print images of the produced map+scenarios

        Run Expert:
            collect agents schedule in the expert solution
            collect interesting metrics

        Generate Dataset:
            group map, scenario and expert solution
            split into train, valid and test set
    """

    # get config from yaml file
    config = get_config_from_yaml("Dataset_creation")

    # create folder for the dataset
    dataset_dir = os.path.join(config.data_root,
                               'random_grid',
                               f'{config.map_size[0]}x{config.map_size[1]}map',
                               f'{config.map_density}density',
                               f'{config.agent_number}agents_{config.task_number}tasks',
                               f'{config.start_position_mode}_start+{config.task_creation_mode}_task')
    create_dirs([dataset_dir])  # if not there, create

    # instantiate creators
    map_creator = MapCreator(map_size=config.map_size,
                             map_density=config.map_density)
    sc_creator = ScenarioCreator(agent_num=config.agent_number)

    # generate map and scenario
    try:
        map_id = 0
        for _ in range(config.map_number):
            # get map
            random_grid_map = map_creator.create_random_grid_map()

            sc_id = 0
            for _ in range(config.scenario_number):
                # get starting positions
                start_pos_list = sc_creator.create_starting_pos(input_map=random_grid_map,
                                                                mode=config.start_position_mode,
                                                                fixed_pos_list=config.fixed_position_list)

                task_list = []
                for _ in range(config.task_number):
                    # get tasks
                    task_list.append(create_task(input_map=random_grid_map,
                                                 mode=config.task_creation_mode,
                                                 start_pos=start_pos_list,
                                                 task_list=task_list))  # list of tasks

                # save the image of map + starting position
                save_image(dataset_dir=dataset_dir,
                           input_map=random_grid_map,
                           map_id=map_id,
                           start_pos_list=start_pos_list,
                           sc_id=sc_id)

                sc_id += 1
            map_id += 1

    except ValueError as err:
        logging.getLogger().warning(err)
        exit(-1)


def create_dataset(map_type):
    """
    Check command line argument and decide which creation function to call
    :param map_type: string, options: 'random_grid'
    """

    if map_type == 'random_grid':
        __random_grid_dataset()


if __name__ == '__main__':
    __spec__ = None
    create_dataset(sys.argv[1])
