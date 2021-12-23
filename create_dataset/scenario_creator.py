"""
Dataset creation utility file
Functions and classes to generate scenarios inside a map
Scenario:
    agents starting positions
    set of tasks
Task:
    pickup and delivery positions

This file does not provide methods to feed tasks to the agents
Tasks are provided in the scenario as a list, how those are passed into the problem is managed externally
It is possible to add tasks into a map/scenario after its creation

Tasks locations can coincide or not with agents starting position or with other tasks locations
Both of these options are controllable with a parameter
"""

from random import sample

import numpy as np


def __extract_task(coord_list):
    """
    Get a task from a list of available coordinates
    :param coord_list: list of tuples (x,y) of available coords
    :return: task ((x,y),(x,y)) -> ((pickup_pos),(delivery_pos))
    """
    idx_list = sample(range(len(coord_list)), k=2)
    pickup_coord = coord_list[idx_list[0]]
    delivery_coord = coord_list[idx_list[1]]

    return pickup_coord, delivery_coord


def create_task(input_map, mode='no_start_repetition',
                start_pos=None, task_list=None):
    """
    Return a task for the given map and starting positions
    Whether tasks can coincide with starting locations or with other tasks is controlled by 'mode'
    :param input_map: np.ndarray, size:H*W, matrix of 0 and 1
    :param mode:
                'free'
                    no restriction on new task position (obviously avoiding obstacles)
                'no_start_repetition'
                    avoid placing new task on agents starting positions
                'no_task_repetition'
                    avoid placing new task on another 'active' task position
                'avoid_all'
                    both 'no_start_repetition' and 'no_task_repetition'
    :param start_pos: list of int tuples, (x,y) -> starting positions of agents
                      passing 'start_pos' only makes sense when 'mode' = 'no_start_repetition' or 'avoid_all'
    :param task_list: list of tuples ((pickup_pos),(delivery_pos)) of current 'active' tasks
                      pickup_pos can be = None, meaning it is available for a new task
                      passing 'task_list' only makes sense when 'mode' = 'no_task_repetition' or 'avoid_all'
    :return: tuple ((x,y),(x,y)) -> ((pickup_pos),(delivery_pos))
    """
    # copy to avoid modifications
    copy_map = input_map.copy()
    # no restrictions
    if mode == 'free':
        # filters out obstacles coords
        where_res = np.nonzero(copy_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))
        # get the task
        task = __extract_task(coord_list=free_cell_list)

    # no tasks on agents starting positions
    elif mode == 'no_start_repetition':
        # set all starting position = 1 in the map
        for pos in start_pos:
            copy_map[pos] = 2
        # filters out obstacles coords and of starting positions
        where_res = np.nonzero(copy_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))
        # get the task
        task = __extract_task(coord_list=free_cell_list)

    # no tasks on other active tasks positions
    elif mode == 'no_task_repetition' or mode == 'avoid_all':

        # avoid both starting positions and other active tasks
        if mode == 'avoid_all':
            # set all starting position = 2 in the map
            for pos in start_pos:
                copy_map[pos] = 2

        # set all other active tasks locations to 1
        for pickup, delivery in task_list:
            if pickup:                  # pickup location can be None (package picked up but not delivered)
                copy_map[pickup] = 2
            copy_map[delivery] = 2        # delivery is always not None, else the task is not active
        # filters out obstacles coords, other active tasks locations and if 'avoid all' also start pos
        where_res = np.nonzero(copy_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))
        # get the task
        task = __extract_task(coord_list=free_cell_list)

    # default: error
    else:
        raise ValueError('Invalid task selection mode')

    return task


class ScenarioCreator:
    """
    Class for creating scenarios inside a given map
    x -> number of the row, y -> number of the column
    Scenario:
        agents starting positions -> [(x0,y0), (x1,y1), ...]
        list of tasks -> [(task1), (task2), ...]
    Task:
        pickup and delivery positions, tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))

    Outside entry points -> create_scenario() for setting up a scenario with N task
                            get_task() for retrieving a new task for a given map
    """

    def __init__(self, agent_num=10):
        """
        Instantiate the class
        :param agent_num: int, number of agents inside the scenario
        """
        self.__agent_num = agent_num

    def create_starting_pos(self, input_map, mode='random', fixed_pos_list=None):
        """
        Get starting position, one for each agent, and collect them
        :param input_map: np.ndarray, size:H*W, matrix of 0 and 1
        :param mode:
                    'random': randomly generates starting positions amongst free cell
                              guarantees that starting positions do not coincide with obstacles
                    'fixed': choose amongst predefined, passed set of positions
                             no guarantees that fixed spots will overwrite obstacles positions
        :param fixed_pos_list: list of int tuples, (x,y), fixed starting positions to choose from
                               only used when 'start_pos_mode' = 'fixed'
        :return: start_pos: list of int tuples, (x,y) -> starting positions of agents
        """
        # copy to avoid modifications
        copy_map = input_map.copy()
        # random generation
        if mode == 'random':
            # flatten the map, get position of 0s
            flat_map = copy_map.flatten()
            zeroes_idx_list = [i for i, x in np.ndenumerate(flat_map)
                               if x == 0
                               ]
            # random permutation of the index list
            p = np.random.permutation(zeroes_idx_list)
            # set selected indexes to 2, reshape as matrix
            flat_map[p[:self.__agent_num]] = 2
            start_map = flat_map.reshape(copy_map.shape)
            # collect starting coordinates in a list
            where_res = np.nonzero(start_map == 2)
            start_pos_list = list(zip(where_res[0], where_res[1]))

        # fixed generation
        elif mode == 'fixed':
            if self.__agent_num > len(fixed_pos_list):
                raise ValueError('Not enough starting positions for all the agents')
            p = np.random.permutation(len(fixed_pos_list))    # random permutation of the idx of pos list
            start_pos_list = [tuple(fixed_pos_list[idx])        # get corresponding starting position
                              for idx in p[:self.__agent_num]   # get indexes out of the permutation
                              ]

        # default: error
        else:
            raise ValueError('Invalid starting position mode')

        return start_pos_list
