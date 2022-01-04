"""
### If you are searching for main of dataset creation, look for GaTp/create_dataset/dataset_creator ###
### Parameters for environment and dataset creation are specified in GaTp/yaml_configs/Dataset_creation.yaml ###

Functions to run an expert over different existing environments
Results are dumped into a pickle file
A matrix-form notation is used to represent produced agent schedule:
    matrix -> 5 (5 actions, z-axis) x num_agent (x-axis) x makespan (max path length, y-axis)
    5 actions order: go_up, go_left, go_down, go_right, stay_still
"""

import pickle
from abc import abstractmethod
from multiprocessing import Pool

from easydict import EasyDict
from gevent import Timeout

from experts.token_passing import tp
from utils.expert_utils import transform_agent_schedule
from utils.file_utils import get_all_files, dump_data


def run_expert(config, dataset_dir):
    """
    Run selected expert to generate solutions for all pre-generated environments
    Expert type supported: 'tp'
    :param config: Namespace of dataset configurations
    :param dataset_dir: path to the dataset directory
    """
    # get path of all file in the dataset dir
    file_path_list = get_all_files(directory=dataset_dir)

    # check out for expert presence
    # if keep_expert = True, return, since nothing to do
    if (any(config.expert_type in file_path for file_path in file_path_list)
            and config.keep_expert_solutions):
        return

    # choose expert type to run
    if config.expert_type == 'tp':
        worker = __TpWorker(config=config)
    else:
        raise ValueError('Invalid expert selected')

    # filter out .png and 'experts' files
    file_path_list = [file_path
                      for file_path in file_path_list
                      if not file_path.endswith('.png')  # filters out images
                      and 'sol' not in file_path]  # filters out expert solutions already there

    # extract with pickle all environments
    env_list = []
    for file_path in file_path_list:
        with open(file_path, 'rb') as f:
            env_list.append(EasyDict(pickle.load(f)))

    # run pool of processes over various environment
    with Pool() as pool:
        # noinspection PyTypeChecker
        pool.map(func=worker, iterable=env_list)        # num of processes == num of cpu processors


class __ExpertWorker:
    """
    Base expert class to run on a process
    """

    def __init__(self, config):
        """
        :param config: Namespace of dataset configurations
        """
        self.config = config

    @abstractmethod
    def __call__(self, environment):
        """
        :param environment: Namespace
                {'name': file path to the env data file,
                 'map': created map,
                 'start_pos_list: agents starting positions,
                 'parking_spot_list': extra non task related endpoints (excluding agents starting points),
                 'task_list': task list built over the map}
        """
        pass


class __TpWorker(__ExpertWorker):
    """
    Worker for token passing, implement ExpertWorker class
    """

    def __call__(self, environment):
        # set up a timer
        timer = Timeout(seconds=self.config.timeout, exception=TimeoutError)
        timer.start()
        try:
            # run token passing
            agent_schedule = tp(input_map=environment.map,
                                start_pos_list=environment.start_pos_list,
                                task_list=environment.task_list,
                                parking_spot_list=environment.parking_spot_list,
                                imm_task_split=self.config.imm_task_split,
                                new_task_per_timestep=self.config.new_task_per_timestep)

            # convert agent schedule into matrix notation
            matrix_schedule = transform_agent_schedule(agent_schedule=agent_schedule)

            # organize data to dump
            file_name = f'{environment.name}_tp_sol'
            expert_data = {'name': file_name,
                           'makespan': matrix_schedule.shape[2],
                           'schedule': matrix_schedule}
            # dump data into pickle file
            dump_data(file_path=file_name, data=expert_data)

        # bad MAPD instance that can not be solved is caught here
        except TimeoutError:
            print('do something')
            # TODO: handle not well formed instances

        # close timer
        finally:
            timer.close()
