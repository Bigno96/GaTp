"""
### If you are searching for main of dataset creation, look for GaTp/create_dataset/dataset_creator.py ###
### Parameters for environment and dataset creation are specified in GaTp/yaml_configs/dataset_creation.yaml ###

Functions to create a data pool of environments, meaning different maps and all the scenarios for those maps

environment = {'name': file path to the env data file,
               'map': created map,
               'start_pos_list': agents starting positions,
               'parking_spot_list': extra non task related endpoints (excluding agents starting points),
               'task_list': task list built over the map}

After creating a data pool, an expert will be run on it to generate the complete dataset
"""
import os
import pickle
from p_tqdm import p_map

from create_dataset.map_creator import create_random_grid_map
from create_dataset.scenario_creator import create_scenario
from utils.file_utils import create_dirs, FolderSwitch
from utils.file_utils import save_image, dump_data


def create_environment(config, dataset_dir, file_path_list=None, recovery_mode=False):
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
    :param file_path_list: list of file path containing environment data to regenerate
                           Pass this ONLY with recovery_mode = True
    :param recovery_mode: boolean, True if create_env is used to re-compute bad MAPD instances
    """
    assert config.map_type in ['random_grid']

    # check out for environment presence
    # if keep_environment = True, return, since nothing to do
    if not recovery_mode and os.path.isdir(dataset_dir) and config.keep_env_data:
        return

    # create directories, if not there
    create_dirs([dataset_dir])

    if not recovery_mode:
        print('Creating Environments')
        # dict to simulate switch case for test/train/valid folders
        folder_switch = FolderSwitch(dataset_dir=dataset_dir, config=config)

        # setup multiprocessing
        worker = EnvironmentWorker(config=config, folder_switch=folder_switch)

        # run pool of processes over map_id sequence
        p_map(worker, range(config.map_number))

    # recovery mode
    else:
        print('Regenerating failed Environments')
        # setup multiprocessing
        worker = RecoveryWorker(config=config)

        # run pool of processes bad file paths
        p_map(worker, file_path_list)


class EnvironmentWorker:
    """
    Worker for creating environment, executed on a thread
    """

    def __init__(self, config, folder_switch):
        """
        :param config: Namespace of dataset configurations
        :param folder_switch: instance of FolderSwitch, simulate switch for saving into folders
        """
        self.config = config
        self.folder_switch = folder_switch

        # pick a map creator
        if self.config.map_type == 'random_grid':
            self.create_map = create_random_grid_map

    def __call__(self, map_id):
        """
        :param map_id: int, id of the map processed
        """
        # get map
        input_map = self.create_map(map_shape=self.config.map_shape,
                                    map_density=self.config.map_density,
                                    connected=self.config.force_conn)

        for sc_id in range(self.config.scenario_number):
            # get scenario
            start_pos_list, parking_spot_list, task_list = create_scenario(config=self.config,
                                                                           input_map=input_map)

            # get directory where to save scenario and map data
            save_dir = self.folder_switch.get_folder(map_id=map_id, sc_id=sc_id)
            file_path = os.path.join(save_dir, f'map{map_id:03d}_case{sc_id:02d}')

            # save map image and dump env file
            save_and_dump(file_path=file_path, input_map=input_map,
                          start_pos_list=start_pos_list, parking_spot_list=parking_spot_list,
                          task_list=task_list)


class RecoveryWorker:
    """
    Worker for recovery mode
    """

    def __init__(self, config):
        """
        :param config: Namespace of dataset configurations
        """
        self.config = config

        # pick a map creator
        if self.config.map_type == 'random_grid':
            self.create_map = create_random_grid_map

    def __call__(self, file_path):
        """
        :param file_path: file path of the bad instance to regenerate
        """
        # get map
        with open(file_path, 'rb') as f:
            environment = pickle.load(f)
        input_map = environment['map']

        # get new scenario over same map
        start_pos_list, parking_spot_list, task_list = create_scenario(config=self.config,
                                                                       input_map=input_map)

        # save map image and dump env file
        save_and_dump(file_path=file_path, input_map=input_map,
                      start_pos_list=start_pos_list, parking_spot_list=parking_spot_list,
                      task_list=task_list)


def save_and_dump(file_path, input_map, start_pos_list, parking_spot_list, task_list):
    # save the image of map + starting position
    save_image(file_path=file_path,
               input_map=input_map,
               start_pos_list=start_pos_list)

    # dump data into a file
    env = {'name': file_path,
           'map': input_map,
           'start_pos_list': start_pos_list,
           'parking_spot_list': parking_spot_list,
           'task_list': task_list}
    dump_data(file_path=file_path, data=env)
