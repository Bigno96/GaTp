"""
Main execution file

-Parse the arguments passed to main
-Process the json configs
-Create agents instances
-Run agents
"""

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
    # positional argument, always first
    arg_parser.add_argument('config_name', type=str,
                            help='File name of configuration in json format')
    # type of execution mode, mandatory
    arg_parser.add_argument('-mode', type=str,
                            help='Train or test mode')

    # environment configurations, if omitted -> default values
    arg_parser.add_argument('-map_type', type=str, default='random_grid',
                            help='Type of map. Options: random_grid')
    arg_parser.add_argument('-map_size', type=int, nargs=2, default=[20, 20],
                            help='Size of the squared map, HxW')
    arg_parser.add_argument('-map_density', type=float, default=0.1,
                            help='Proportion of occupied over free space in the environment')
    arg_parser.add_argument('-num_agents', type=int, default=20,
                            help='Number of agents in the map')
    arg_parser.add_argument('-tasks_number', type=int, default=500,
                            help='Number of tasks that will be fed to the agents')
    arg_parser.add_argument('-imm_task_split', type=float, default=0.0,
                            help='Percentage of tasks immediately available at the start')
    arg_parser.add_argument('-new_task_per_timestep', type=int, default=1,
                            help='How many tasks to add at each insertion')
    arg_parser.add_argument('-step_between_insertion', type=int, default=1,
                            help='How many timesteps pass between each insertion')
    arg_parser.add_argument('-start_position_mode', type=str, default='random',
                            help='How starting positions were created. Options: random or fixed')
    arg_parser.add_argument('-task_creation_mode', type=str, default='avoid_non_task_rep',
                            help='How tasks were created. Options: free, avoid_non_task_rep, avoid_task_rep or avoid_all')

    # get the argument from the console
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args)


if __name__ == '__main__':
    __spec__ = None
    main()
