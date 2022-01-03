"""
### If you are searching for main of dataset creation, look for GaTp/create_dataset/dataset_creator ###
### Parameters for environment and dataset creation are specified in GaTp/yaml_configs/Dataset_creation.yaml ###

Functions to create a data pool
Data pools consist of maps and all the scenarios for those maps
After creating a data pool, an expert will be run on it to generate the complete dataset
"""
import logging
import os

from create_dataset.map_creator import *
from create_dataset.scenario_creator import *
from utils.file_utils import create_folder_switch, get_folder_from_switch
from utils.file_utils import save_image, dump_data


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
    folder_switcher = create_folder_switch(dataset_dir=dataset_dir, config=config)
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
                save_dir = get_folder_from_switch(map_id=map_id, switcher_dict=folder_switcher)
                file_path = os.path.join(save_dir, f'map{map_id:03d}_case{sc_id:02d}')

                # save the image of map + starting position
                save_image(file_path=file_path,
                           input_map=random_grid_map,
                           start_pos_list=start_pos_list)

                # dump data into a file
                data = {'map': random_grid_map,
                        'non_task_ep_list': non_task_ep_list,
                        'task_list': task_list}
                dump_data(file_path=file_path, data=data)

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
