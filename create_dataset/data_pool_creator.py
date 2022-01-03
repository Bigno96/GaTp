"""
### If you are searching for main of dataset creation, look for GaTp/create_dataset/dataset_creator ###
### Parameters for environment and dataset creation are specified in GaTp/yaml_configs/Dataset_creation.yaml ###

Functions to create a data pool
Data pools consist of maps and all the scenarios for those maps
After creating a data pool, an expert will be run on it to generate the complete dataset
"""
import logging
import os

from PIL import Image
from PIL.ImageOps import invert, colorize

from create_dataset.map_creator import *
from create_dataset.scenario_creator import *
from utils.create_dirs import create_dirs


def __get_folder(map_id, switcher_dict):
    """
    Simulate a switch-case to decide which folder to use
    :param map_id: id of the map on which scenarios are being created
    :param switcher_dict: dict -> {range : folder}
    :return: folder
    """
    for key, value in switcher_dict.items():
        if map_id in key:
            return value

    raise ValueError('Map ID out of switcher range')


def save_image(dataset_dir, input_map, map_id, start_pos_list, sc_id):
    """
    Save image of the map (black and white) with agent starting positions (red)
    :param dataset_dir: path to the specific dataset directory
    :param input_map: np.ndarray, size: H*W, matrix of '0' and '1'
    :param map_id: int, identifier of map
    :param start_pos_list: list of tuples, (x,y) -> coordinates over the map
    :param sc_id: int, identifier of scenario
    """
    img_path = os.path.join(dataset_dir, f'map{map_id:03d}_case{sc_id:02d}.png')
    # input map with float values
    f_map = np.array(input_map, dtype=float)
    for pos in start_pos_list:
        f_map[pos] = 0.5
    # white -> background | black -> obstacles | red -> agents starting positions
    colorize(invert(Image.fromarray(obj=np.uint8(f_map*255))),
             black='black', white='white', mid='red').save(img_path)


def __random_grid_data_pool(config, dataset_dir):
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

        Split into train, valid and test set

    :param config: Namespace of dataset configurations
    :param dataset_dir: path to the dataset directory
    """
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

    create_dirs([train_dir, valid_dir, test_dir])  # if not there, create

    # set how to divide scenarios between train, validation and test
    train_split = int(config.train_split * config.map_number)
    valid_split = int(config.valid_split * config.map_number)

    # dictionary to emulate a switch case with ranges
    folder_switcher = {
        range(0, train_split): train_dir,
        range(train_split, train_split+valid_split): valid_dir,
        range(train_split+valid_split, config.map_number): test_dir
    }

    # generate map and scenario
    try:
        for map_id in range(config.map_number):
            # get map
            random_grid_map = create_random_grid_map(map_shape=config.map_shape,
                                                     map_density=config.map_density)

            for sc_id in range(config.scenario_number):
                # get starting positions
                start_pos_list = create_starting_pos(input_map=random_grid_map,
                                                     agent_num=config.agent_number,
                                                     mode=config.start_position_mode,
                                                     fixed_pos_list=config.fixed_position_list)

                # non task endpoints list
                non_task_ep_list = start_pos_list.copy()

                # get task list
                task_list = []
                for _ in range(config.task_number):
                    task_list.append(create_task(input_map=random_grid_map,
                                                 mode=config.task_creation_mode,
                                                 non_task_ep_list=non_task_ep_list,
                                                 task_list=task_list))  # list of tasks, passed recursively

                # get directory where to save scenario and map data
                save_dir = __get_folder(map_id=map_id, switcher_dict=folder_switcher)

                # save the image of map + starting position
                save_image(dataset_dir=save_dir,
                           input_map=random_grid_map,
                           map_id=map_id,
                           start_pos_list=start_pos_list,
                           sc_id=sc_id)

                # TODO: dump into a matlab file map and scenario data

    # invalid configuration parameters passed
    except ValueError as err:
        logging.getLogger().warning(err)
        exit(-1)


def create_data_pool(config, dataset_dir):
    """
    Check command line argument and decide which creation function to call
    :param config: Namespace of dataset configurations
    :param dataset_dir: path to the dataset directory
    """

    if config.map_type == 'random_grid':
        __random_grid_data_pool(config=config, dataset_dir=dataset_dir)
