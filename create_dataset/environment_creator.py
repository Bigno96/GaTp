"""
### If you are searching for main of dataset creation, look for GaTp/create_dataset/dataset_creator ###
### Parameters for environment and dataset creation are specified in GaTp/yaml_configs/Dataset_creation.yaml ###

Functions to create a data pool of environments, meaning different maps and all the scenarios for those maps

environment = {'name': file path to the env data file,
               'map': created map,
               'start_pos_list: agents starting positions,
               'parking_spot_list': extra non task related endpoints (excluding agents starting points),
               'task_list': task list built over the map}

After creating a data pool, an expert will be run on it to generate the complete dataset
"""
import os
from multiprocessing import Pool

from create_dataset.map_creator import *
from create_dataset.scenario_creator import *
from utils.file_utils import create_dirs
from utils.file_utils import create_folder_switch, get_folder_from_switch
from utils.file_utils import save_image, dump_data


def create_environment(config, dataset_dir):
    """
    Check config.map_type and decide which creation function to call
    :param config: Namespace of dataset configurations
    :param dataset_dir: path to the dataset directory
    """

    # check out for environment presence
    # if keep_environment = True, return, since nothing to do
    if os.path.isdir(dataset_dir) and config.keep_env_data:
        return

    # create directories, if not there
    create_dirs([dataset_dir])

    if config.map_type == 'random_grid':
        __random_grid_environment(config=config, dataset_dir=dataset_dir)
    else:
        raise ValueError('Invalid map type selected')


def __random_grid_environment(config, dataset_dir):
    """
    Pipeline:
        Create Maps (see map_creator.py for more details):
            MAP_NUMBER different random grid maps for the same setting (size, density)

        Create Scenarios (see scenario_creator.py for more details):
            To have higher probability of well-formed MAPD instances,
            create_task should be set with mode = 'no_start_repetition'
            This will guarantee a set of non-task endpoints >= num agents, being the agents' starting positions

            SCENARIO_NUMBER different scenarios for each map
            AGENT_NUMBER different starting positions for agents
            TASK_NUMBER series of tasks (pickup and delivery positions)

            save images of the produced map + starting points
            save data of the produced environment (map + scenario)

        Splitting various environments into train, valid and test set

    :param config: Namespace of dataset configurations
    :param dataset_dir: path to the dataset directory
    """
    # dict to simulate switch case for test/train/valid folders
    folder_switcher = create_folder_switch(dataset_dir=dataset_dir, config=config)

    # setup multiprocessing
    worker = __RandomGridWorker(config=config, folder_switcher=folder_switcher)

    # run pool of processes over map_id sequence
    with Pool() as pool:
        pool.map(func=worker, iterable=range(config.map_number))


class __RandomGridWorker:
    """
    Worker for __random_grid_environment, executed on a thread
    """

    def __init__(self, config, folder_switcher):
        """
        :param config: Namespace of dataset configurations
        :param folder_switcher: dict, containing folder to save into
        """
        self.config = config
        self.folder_switcher = folder_switcher

    def __call__(self, map_id):
        """
        :param map_id: int, id of the map processed
        """
        # get map
        random_grid_map = create_random_grid_map(map_shape=self.config.map_shape,
                                                 map_density=self.config.map_density)

        for sc_id in range(self.config.scenario_number):
            # get starting positions
            start_pos_list = create_starting_pos(input_map=random_grid_map,
                                                 agent_num=self.config.agent_number,
                                                 mode=self.config.start_position_mode,
                                                 fixed_pos_list=self.config.fixed_position_list)

            # non task endpoints list
            parking_spot_list = []      # should not contain any agent starting position
            non_task_ep_list = start_pos_list + parking_spot_list

            # get task list
            task_list = []
            for _ in range(self.config.task_number):
                task_list.append(create_task(input_map=random_grid_map,
                                             mode=self.config.task_creation_mode,
                                             non_task_ep_list=non_task_ep_list,
                                             task_list=task_list))  # list of tasks, passed recursively

            # get directory where to save scenario and map data
            save_dir = get_folder_from_switch(map_id=map_id, switcher_dict=self.folder_switcher)
            file_path = os.path.join(save_dir, f'map{map_id:03d}_case{sc_id:02d}')

            # save the image of map + starting position
            save_image(file_path=file_path,
                       input_map=random_grid_map,
                       start_pos_list=start_pos_list)

            # dump data into a file
            env = {'name': file_path,
                   'map': random_grid_map,
                   'start_pos_list': start_pos_list,
                   'parking_spot_list': parking_spot_list,
                   'task_list': task_list}
            dump_data(file_path=file_path, data=env)
