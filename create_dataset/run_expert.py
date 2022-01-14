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
from multiprocessing import Pool, Manager
from os.path import normpath, basename

from easydict import EasyDict
from gevent import Timeout

from experts.token_passing import tp
from utils.expert_utils import transform_agent_schedule
from utils.file_utils import get_all_files, dump_data
from utils.metrics import count_collision


def run_expert(config, dataset_dir, file_path_list=None, recovery_mode=False):
    """
    Run selected expert to generate solutions for all pre-generated environments
    Expert type supported: 'tp'
    :param config: Namespace of dataset configurations
    :param dataset_dir: path to the dataset directory
    :param file_path_list: list of file path containing environment data to run expert over
                           Pass this ONLY with recovery_mode = True
    :param recovery_mode: boolean, True if run_expert is used to re-compute bad MAPD instances
    :return bad_instances_list, list with the name (map_id + scenario_id) of bad MAPD instance files
    """
    # get path of all file in the dataset dir, if not in recovery mode
    if not recovery_mode:
        file_path_list = get_all_files(directory=dataset_dir)

        # check out for expert presence
        # if keep_expert = True, return, since nothing to do
        if (any(config.expert_type in file_path for file_path in file_path_list)
                and config.keep_expert_solutions):
            return []

        # filter out .png and 'experts' files
        file_path_list = [file_path
                          for file_path in file_path_list
                          if not file_path.endswith('.png')  # filters out images
                          and 'sol' not in file_path]  # filters out expert solutions already there

        # no environment file found
        if not file_path_list:
            raise ValueError('No environment files found')

    # no file paths given while recovery mode
    if not file_path_list:
        raise ValueError('Experts launched in recovery mode with no file paths')

    # extract with pickle all environments
    env_list = []
    for file_path in file_path_list:
        with open(file_path, 'rb') as f:
            env_list.append(EasyDict(pickle.load(f)))

    # get shared list
    manager = Manager()
    bad_instances_list = manager.list()
    # choose expert type to run
    if config.expert_type == 'tp':
        worker = __TpWorker(config=config, bad_instances_list=bad_instances_list)
    else:
        raise ValueError('Invalid expert selected')

    # run pool of processes over various environment
    with Pool() as pool:
        # noinspection PyTypeChecker
        pool.map(func=worker, iterable=env_list)    # num of processes == num of cpu processors

    return list(bad_instances_list)


class __ExpertWorker:
    """
    Base expert class to run on a process
    """

    def __init__(self, config, bad_instances_list):
        """
        :param config: Namespace of dataset configurations
        :param bad_instances_list: manager list, where to write bad environment filenames
        """
        self.config = config
        self.bad_instances_list = bad_instances_list

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
            agent_schedule, \
                service_time, \
                timestep_runtime = tp(input_map=environment.map,
                                      start_pos_list=environment.start_pos_list,
                                      task_list=environment.task_list,
                                      parking_spot_list=environment.parking_spot_list,
                                      imm_task_split=self.config.imm_task_split,
                                      new_task_per_insertion=self.config.new_task_per_timestep,
                                      step_between_insertion=self.config.step_between_insertion)

            collision_count = count_collision(agent_schedule=agent_schedule)

            # no collisions
            if not collision_count:
                # convert agent schedule into matrix notation
                matrix_schedule = transform_agent_schedule(agent_schedule=agent_schedule)

                # organize data to dump
                file_name = f'{environment.name}_tp_sol'
                expert_data = {'name': file_name,
                               'makespan': matrix_schedule.shape[2],
                               'service_time': service_time,
                               'runtime_per_timestep': timestep_runtime,
                               'schedule': matrix_schedule}
                # dump data into pickle file
                dump_data(file_path=file_name, data=expert_data)

                name = basename(normpath(environment.name))
                print(f'Running Expert on Scenario {name}')

            # collisions found
            else:
                self.bad_instances_list.append(environment.name)

        # bad MAPD instance that can not be solved is caught here
        except TimeoutError:
            self.bad_instances_list.append(environment.name)

        # close timer
        finally:
            timer.close()