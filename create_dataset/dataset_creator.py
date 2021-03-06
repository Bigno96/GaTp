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

All parameters are controlled in GaTp/yaml_configs/dataset_creation.yaml

Inside each training, validation and testing folders there are:
- image of the map + starting positions
- file with environment data:
        {'name', 'map', 'start_pos_list, 'parking_spot_list', 'task_list'}
- file with the representation of the agents schedule obtained by the expert (different experts, different files):


## More information at GaTp/create_dataset/environment_creator.py and GaTp/create_dataset/run_expert.py ##
"""

import logging
import os
import create_dataset.environment_creator as env_cr
import create_dataset.run_expert as exp
import create_dataset.nn_data_generator as nn_data_gen
import utils.config as cfg

DATA_ROOT = 'D:/Uni/TESI'


def create_dataset() -> None:
    """
    Create dataset consisting of maps, scenarios and experts solutions
    """
    # get logger
    logger = logging.getLogger("Dataset Creator")
    # get config from yaml file
    config = cfg.get_config_from_yaml('dataset_creation')
    # data root
    config.data_root = os.path.join(DATA_ROOT, config.data_root)

    # create folder for the dataset
    dataset_dir = os.path.join(config.data_root,
                               'random_grid',
                               f'{config.map_shape[0]}x{config.map_shape[1]}map',
                               f'{config.map_density}density',
                               f'{config.agent_number}agents_{config.task_number}tasks_'
                               f'{config.imm_task_split}split_'
                               f'+{config.new_task_per_timestep}_every{config.step_between_insertion}',
                               f'{config.start_position_mode}_start+{config.task_creation_mode}_task')

    try:
        # create maps with scenarios
        env_cr.create_environment(config=config,
                                  dataset_dir=dataset_dir)
        # run expert over those environments
        bad_instances_list = exp.run_expert(config=config,
                                            dataset_dir=dataset_dir)
        # transform data for NN
        nn_data_gen.get_nn_data(config=config,
                                dataset_dir=dataset_dir,
                                bad_instances_list=bad_instances_list)

        bad_instances_count = 0
        # until no bad MAPD instances are left, repeat their generation
        while bad_instances_list:
            print(f'\n\nFound bad MAPD instances')
            bad_instances_count += len(bad_instances_list)
            old_bad_instances = bad_instances_list.copy()   # to pass to get_nn_data
            env_cr.create_environment(config=config,
                                      dataset_dir=dataset_dir,
                                      recovery_mode=True,
                                      file_path_list=bad_instances_list)
            bad_instances_list = exp.run_expert(config=config,
                                                dataset_dir=dataset_dir,
                                                recovery_mode=True,
                                                file_path_list=bad_instances_list)
            nn_data_gen.get_nn_data(config=config,
                                    dataset_dir=dataset_dir,
                                    bad_instances_list=bad_instances_list,
                                    recovery_mode=True,
                                    file_path_list=old_bad_instances)

        print(f'\n\nRegenerated {bad_instances_count} bad MAPD instances')

    # invalid configuration parameters passed
    except ValueError as err:
        logger.warning(err)
        exit(-1)

    # invalid configuration parameters passed
    except AssertionError as err:
        logger.warning(err)
        exit(-1)


if __name__ == '__main__':
    __spec__ = None
    create_dataset()
