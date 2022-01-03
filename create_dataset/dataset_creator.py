import os
from utils.config import get_config_from_yaml
from utils.create_dirs import create_dirs

# get config from yaml file
config = get_config_from_yaml("Dataset_creation")

# create folder for the dataset
dataset_dir = os.path.join(config.data_root,
                           'random_grid',
                           f'{config.map_shape[0]}x{config.map_shape[1]}map',
                           f'{config.map_density}density',
                           f'{config.agent_number}agents_{config.task_number}tasks',
                           f'{config.start_position_mode}_start+{config.task_creation_mode}_task')

create_dirs([dataset_dir])
