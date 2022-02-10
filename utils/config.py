"""
Utility file
Parse file configuration and add them to the input configuration, creating a config Namespace
Set up all the logging for the project
"""

import logging
import os
import pprint
from datetime import datetime
from logging import Formatter
from logging.handlers import RotatingFileHandler
from time import mktime

from easydict import EasyDict
from yaml import safe_load

from utils.file_utils import create_dirs

CONFIG_FOLDER_PATH = 'D:\\Uni\\TESI\\GaTp\\yaml_configs'


def setup_logging(log_dir):
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
    exp_file_handler = RotatingFileHandler(f'{log_dir}\\exp_debug.log', maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    # handler for errors (WARNING level) during execution of experiments
    # dump into a file: log_dir/exp_error.log
    exp_errors_file_handler = RotatingFileHandler(f'{log_dir}\\exp_error.log', maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    # bind handlers to main logger
    if not main_logger.handlers:
        main_logger.addHandler(console_handler)
        main_logger.addHandler(exp_file_handler)
        main_logger.addHandler(exp_errors_file_handler)


def get_config_from_yaml(config_name):
    """
    Get the config from a yaml file and return as namespace
    :param config_name: the name of the config file
    :return: config(namespace)
    """

    # extend name with .yaml and the correct folder
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


def process_config(args):
    """
    Get the yaml file
    Processing it with EasyDict to be accessible as attributes
    Set up the logging in the whole program
    Then return the config
    :param args: argument parser output
    :return: config object(namespace)
    """

    # load yaml with the selected configuration into a Namespace
    config = get_config_from_yaml(args.config_name)

    ''' 
    Add all args, parsed from main.py inputs, as Namespace attributes
    '''
    config.mode = args.mode     # train or test
    # environment configuration
    config.map_type = args.map_type
    config.map_size = args.map_size
    config.map_density = args.map_density
    config.num_agents = args.num_agents
    config.tasks_number = args.tasks_number
    config.imm_task_split = args.imm_task_split
    config.new_task_per_timestep = args.new_task_per_timestep
    config.step_between_insertion = args.step_between_insertion
    config.start_position_mode = args.start_position_mode
    config.task_creation_mode = args.task_creation_mode

    # set up experiment name with configuration summary:
    #   environment description, hyper parameters, timestamp
    config.env_setup = f'{config.map_type}_{config.map_size[0]}x{config.map_size[1]}' \
                       f'_{config.map_density}density_{config.num_agents}agents_{config.tasks_number}'
    config.task_setup = f'{config.start_position_mode}_start+{config.task_creation_mode}_task' \
                        f'+{config.imm_task_split}split_+{config.new_task_per_timestep}' \
                        f'_every{config.step_between_insertion}'
    config.exp_hyper_para = f'{config.attention_heads}heads+{config.attention_concat}_concat' \
                            f'+{config.communication_hops}hops'
    config.exp_time = str(int(mktime(datetime.now().timetuple())))
    config.exp_name = os.path.join(f'{config.model_name}_{config.env_setup}',
                                   config.exp_hyper_para,
                                   config.task_setup,
                                   config.exp_time)

    # setup and create useful directories
    config.log_dir = os.path.join(config.exp_folder,
                                  config.exp_name,
                                  'logs')           # logging outputs
    config.out_dir = os.path.join(config.exp_folder,
                                  config.exp_name,
                                  'out')            # experiment outputs
    create_dirs([config.log_dir, config.out_dir])

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info('Hi, This is root.')
    logging.getLogger().info('The configurations have been successfully processed and directories created.')
    logging.getLogger().info('The pipeline of the project will begin now.')

    # print out processed configuration
    print('Selected configuration is the following:')
    pprint.pprint(config, sort_dicts=False)

    return config
