"""
General utility file for configurations

Parse file configuration and add them to the input configuration, creating a config Namespace
Set up all the logging for the project
"""

import logging
import os
import pprint
import argparse
import sys
import time

import utils.file_utils as f_utils

from datetime import datetime
from logging import Formatter
from logging.handlers import RotatingFileHandler
from time import mktime
from easydict import EasyDict
from yaml import safe_load

CONFIG_FOLDER_PATH = 'GaTp/yaml_configs'
PROJECT_ROOT = 'D:/Uni/TESI'
APPEND_PROJECT_ROOT = True       # change this to prepend PROJECT_ROOT for creating folder paths


def setup_logging(log_dir: str) -> None:
    """
    Set up main logger and handlers for outputting information during execution
    :param log_dir: directory where to create dump file for experiments handlers
    """
    # messages formatting
    log_file_format = '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d'
    log_console_format = '[%(levelname)s]: %(message)s'

    # main logger, level = INFO
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    # handler for console messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    # handler for DEBUG level information during execution of experiments
    # dump into a file: log_dir/exp_debug.log
    exp_file_handler = RotatingFileHandler(f'{log_dir}/exp_debug.log', maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    # handler for errors (WARNING level) during execution of experiments
    # dump into a file: log_dir/exp_error.log
    exp_errors_file_handler = RotatingFileHandler(f'{log_dir}/exp_error.log', maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    # bind handlers to main logger
    if not main_logger.handlers:
        main_logger.addHandler(console_handler)
        main_logger.addHandler(exp_file_handler)
        main_logger.addHandler(exp_errors_file_handler)


def get_config_from_yaml(config_name: str) -> EasyDict:
    """
    Get the config from a yaml file and return as namespace
    :param config_name: the name of the config file
    :return: namespace of the config written in the file
    """

    # extend name with .yaml and the correct folder
    if APPEND_PROJECT_ROOT:
        yaml_file = os.path.join(PROJECT_ROOT, CONFIG_FOLDER_PATH, f'{config_name}.yaml')
    else:
        yaml_file = os.path.join(CONFIG_FOLDER_PATH, f'{config_name}.yaml')

    # parse the configurations from the config yaml file provided
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = safe_load(config_file)
            # EasyDict allows accessing dict values as attributes (works recursively)
            return EasyDict(config_dict)
        except ValueError:
            print('INVALID YAML file format. Please provide a good YAML file')
            exit(-1)


# noinspection DuplicatedCode
def process_config(args: argparse.Namespace) -> EasyDict:
    """
    Get the yaml file
    Processing it with EasyDict to be accessible as attributes
    Set up the logging in the whole program
    Then return the config
    :param args: argument parser output (Namespace)
    :return: config object (Namespace)
    """

    # load yaml with the selected configuration into a Namespace
    config = get_config_from_yaml(args.agent_type)

    ''' 
    Add all args, parsed from main.py inputs, as Namespace attributes
    '''
    # train or test
    config.mode = args.mode
    # agent type
    config.agent_type = args.agent_type
    # skip validation
    config.skip_valid = args.skip_valid
    # dagger algorithm
    config.use_dagger = args.dagger

    if config.use_dagger and config.skip_valid:
        print('When using dagger algorithm, validation cannot be skipped. Validation will be performed.')
        config.skip_valid = False

    if config.use_dagger and config.mode != 'train':
        print('Cannot use dagger outside training mode!')
        exit(-1)

    # prepend data root
    if APPEND_PROJECT_ROOT:
        config.data_root = os.path.join(PROJECT_ROOT, config.data_root)

    # environment configuration
    config.map_type = args.map_type
    config.map_shape = args.map_shape
    config.map_density = args.map_density
    config.agent_number = args.agent_num
    config.task_number = args.task_num
    config.imm_task_split = args.imm_split
    config.new_task_per_timestep = args.new_task_ts
    config.step_between_insertion = args.insertion_step
    config.start_position_mode = args.start_pos_mode
    config.task_creation_mode = args.task_mode
    config.expert_type = args.expert_type

    # agent configuration
    config.transform_runtime_data = args.tf_runtime_data
    config.FOV = args.FOV
    config.comm_radius = args.comm_radius
    config.sim_num_process = args.sim_num_process
    config.load_checkpoint = args.load_ckp
    config.load_ckp_mode = args.load_ckp_mode
    config.epoch_id = args.epoch_id
    config.checkpoint_timestamp = args.ckp_ts

    # set up experiment name with configuration summary:
    #   environment description, hyper parameters, timestamp
    config.env_setup = f'{config.map_type}_{config.map_shape[0]}x{config.map_shape[1]}' \
                       f'_{config.map_density}density_{config.agent_number}agents_{config.task_number}'
    config.task_setup = f'{config.start_position_mode}_start+{config.task_creation_mode}_task' \
                        f'+{config.imm_task_split}split_+{config.new_task_per_timestep}' \
                        f'_every{config.step_between_insertion}'
    config.exp_hyper_para = f'{config.attention_heads}heads+{config.attention_concat}_concat' \
                            f'+{config.comm_radius}comm_radius+{config.FOV}FOV'

    # initial checkpoint loading only for train mode
    # test and valid always load their checkpoints automatically
    config.load_checkpoint = config.load_checkpoint and config.mode == 'train'

    # if loading an existing checkpoint
    if config.load_checkpoint:
        # checkpoint to load not specified
        if not config.checkpoint_timestamp:
            print('Error: No checkpoint to load!')
            exit(-1)
        # check that if load_mode == 'epoch', an epoch is specified
        if config.load_ckp_mode == 'epoch' and config.epoch_id is None:
            print('Error: No epoch specified when trying to load checkpoint!')
            exit(-1)

        config.exp_time = config.checkpoint_timestamp

    # test and valid only modes do load model checkpoints
    elif config.mode == 'test' or config.mode == 'valid':
        # checkpoint to load not specified
        if not config.checkpoint_timestamp:
            print('Error: No checkpoint to load!')
            exit(-1)

        config.exp_time = config.checkpoint_timestamp

    # not loading a pre-existing checkpoint
    else:
        config.exp_time = str(int(mktime(datetime.now().timetuple())))

    # exp folder path
    config.exp_name = os.path.join(f'{config.agent_type.upper()}_{config.env_setup}',
                                   config.exp_hyper_para,
                                   config.task_setup,
                                   config.exp_time)

    # setup useful directories
    if APPEND_PROJECT_ROOT:
        config.exp_folder = os.path.join(PROJECT_ROOT, config.exp_folder)
    config.log_dir = os.path.join(config.exp_folder,
                                  config.exp_name,
                                  'logs')   # logging outputs
    config.checkpoint_dir = os.path.join(config.exp_folder,
                                         config.exp_name,
                                         'checkpoints')     # checkpoint folder
    # create, if they don't exist
    f_utils.create_dirs([config.log_dir, config.checkpoint_dir])

    # Windows does not support multiprocess pytorch loading
    config.data_loader_workers = 0 if os.name == 'nt' else config.data_loader_workers

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info('Hi, This is root.')
    logging.getLogger().info('The configurations have been successfully processed and directories created.')
    logging.getLogger().info('The pipeline of the project will begin now.')

    # print out processed configuration
    time.sleep(1)  # aligning prints
    print('Selected configuration is the following:')
    pprint.pprint(config)
    sys.stdout.flush()

    time.sleep(1)   # aligning prints
    return config
