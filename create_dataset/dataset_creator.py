"""
Run to create a dataset
Command line argument:
    - random_grid: create random_grid dataset
Parameters for the environment and the dataset are specified as constants
"""
import logging
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from map_creator import *
from scenario_creator import *
from utils.create_dirs import create_dirs

# datasets root folder
DATA_ROOT = 'D:\\Uni\\TESI\\GaTp\\datasets'

'''
Map specifications
'''
MAP_NUMBER = 1000       # how many unique maps
MAP_SIZE = (20, 20)     # (H, W)
MAP_DENSITY = 0.2       # % of obstacles in the map

'''
Scenario specifications
'''
SCENARIO_NUMBER = 50                # how many unique scenarios for each map
AGENT_NUMBER = 10                   # how many agents in each scenario
TASK_NUMBER = 100                   # how many tasks in each scenario
# how to create starting positions for agents. 'random' or 'fixed'
START_POSITION_MODE = 'random'
# if 'fixed', give list len([(x,y)]) >= num_agents of starting pos to choose from
FIXED_POSITION_LIST = [(1, 1), (1, 2), (1, 3), (1, 4)]
# how to create tasks. 'free', 'no_start_repetition', 'no_task_repetition' or 'avoid_all'
TASK_CREATION_MODE = 'no_task_repetition'


def save_image(dataset_dir, input_map, map_id, start_pos, sc_id):
    """
    Save image of the map (black and white) with agent starting positions (red)
    :param dataset_dir: path to the specific dataset directory
    :param input_map: np.ndarray, size: H*W, matrix of '0' and '1'
    :param map_id: int, identifier of map
    :param start_pos: list of tuples, (x,y) -> coordinates over the map
    :param sc_id: int, identifier of scenario
    """
    img_path = os.path.join(dataset_dir, f'map{map_id:03d}_case{sc_id:02d}.png')
    for pos in start_pos:
        input_map[pos] = 2
    # white = 0, background | black = 1, obstacles | red = 2, agents starting positions
    plt.imsave(img_path, input_map,
               cmap=ListedColormap(['white', 'black', 'red']))


def create_dataset(map_type):
    """
    Check command line argument and decide which creation function to call
    :param map_type: string, options: 'random_grid'
    """

    if map_type == 'random_grid':
        __random_grid_dataset()


def __random_grid_dataset():
    """
    Pipeline:
        Create Maps:
            MAP_NUMBER different random grid maps for the same setting (size, density)
        Create Scenarios:
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

    map_id = 0
    sc_id = 0

    # create folder for the dataset
    dataset_dir = os.path.join(DATA_ROOT,
                               'random_grid',
                               f'{MAP_SIZE[0]}x{MAP_SIZE[1]}map',
                               f'{MAP_DENSITY}density',
                               f'{AGENT_NUMBER}agents_{TASK_NUMBER}tasks')
    create_dirs([dataset_dir])      # if not there, create

    # instantiate creators
    map_creator = MapCreator(map_size=MAP_SIZE,
                             map_density=MAP_DENSITY)
    sc_creator = ScenarioCreator(agent_num=AGENT_NUMBER)

    # generate map and scenario
    random_grid_map = None          # map with obstacles
    start_pos_list = []             # list of agents starting positions
    task_list = []                  # list of tasks
    try:
        # get map
        random_grid_map = map_creator.create_random_grid_map()
        # get starting positions
        start_pos_list = sc_creator.create_starting_pos(input_map=random_grid_map,
                                                        mode=START_POSITION_MODE,
                                                        fixed_pos_list=FIXED_POSITION_LIST)
        # get tasks
        task_list = [create_task(input_map=random_grid_map,
                                 mode=TASK_CREATION_MODE,
                                 start_pos=start_pos_list,
                                 task_list=task_list)
                     for _ in range(TASK_NUMBER)]

    except ValueError as err:
        logging.getLogger().warning(err)
        exit(-1)

    # save the image of map + starting position
    save_image(dataset_dir=dataset_dir,
               input_map=random_grid_map,
               map_id=map_id,
               start_pos=start_pos_list,
               sc_id=sc_id)


if __name__ == '__main__':
    __spec__ = None
    create_dataset(sys.argv[1])
