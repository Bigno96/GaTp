"""
Main execution file

-Parse the arguments passed to main
-Process the json configs
-Create agents instances
-Run agents
"""
import argparse
from argparse import ArgumentParser

from utils.config import process_config


def main():
    """
    Main function
    Parses argument
    Create and run agents
    """

    # set up argument parser to read input arguments
    arg_parser = ArgumentParser(description="")

    '''
    See README.md for detailed explanations
    '''
    # mandatory arguments
    arg_parser.add_argument('-agent_type', type=str, required=True,
                            choices=['magat'],
                            help='Type of agent to deploy')
    arg_parser.add_argument('-mode', type=str, required=True,
                            choices=['train', 'test'],
                            help='Train or test mode')

    # environment parameters, if omitted -> default values
    arg_parser.add_argument('-map_type', type=str, default='random_grid',
                            choices=['random_grid'],
                            help='Type of map')
    arg_parser.add_argument('-map_shape', type=int, nargs=2, default=[20, 20],
                            help='Shape of the squared map, HxW')
    arg_parser.add_argument('-map_density', type=float, default=0.1,
                            help='Proportion of occupied over free space in the environment')
    arg_parser.add_argument('-agent_number', type=int, default=20,
                            help='Number of agents in the map')
    arg_parser.add_argument('-task_number', type=int, default=500,
                            help='Number of tasks that will be fed to the agents')
    arg_parser.add_argument('-imm_task_split', type=float, default=0.0,
                            help='Percentage of tasks immediately available at the start')
    arg_parser.add_argument('-new_task_per_timestep', type=int, default=1,
                            help='How many tasks to add at each insertion')
    arg_parser.add_argument('-step_between_insertion', type=int, default=1,
                            help='How many timesteps pass between each insertion')
    arg_parser.add_argument('-start_position_mode', type=str, default='random',
                            choices=['random', 'fixed'],
                            help='How starting positions were created')
    arg_parser.add_argument('-task_creation_mode', type=str, default='avoid_non_task_rep',
                            choices=['free', 'avoid_non_task_rep', 'avoid_task_rep', 'avoid_all'],
                            help='How tasks were created')
    arg_parser.add_argument('-expert_type', type=str, default='tp',
                            choices=['tp'],
                            help='Expert used for solving cases')

    # agent parameters, if omitted -> default values
    arg_parser.add_argument('-FOV', type=__check_odd, default=9,
                            help='Radius of agents FOV. Has to be odd')
    arg_parser.add_argument('-comm_radius', type=int, default=7,
                            help='Maximum communication distance between agents')
    arg_parser.add_argument('-sim_num_process', type=int, default=4,
                            help='Number of separate processes for running agent simulation')
    arg_parser.add_argument('-load_checkpoint', action='store_true',
                            help='Add this flag if you want to load a pretrained checkpoint')
    arg_parser.add_argument('-load_ckp_mode', type=str, default='latest',
                            choices=['latest', 'best', 'epoch'],
                            help='Mode of loading checkpoint: latest, best or specific epoch')
    arg_parser.add_argument('-load_epoch', type=int, default=None,
                            help='Number of epoch to load if load_ckp_mode = epoch')

    # MANDATORY if config.mode == 'test' or -load_checkpoint
    arg_parser.add_argument('-checkpoint_timestamp', type=str,  default=None,
                            help='Timestamp (folder name) of the checkpoint to load')

    # get the argument from the console
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args)


def __check_odd(v):
    if int(v) % 2 == 0:
        raise argparse.ArgumentTypeError('not an odd number')
    return int(v)


if __name__ == '__main__':
    __spec__ = None
    main()
