"""
Dataset creation utility file
Functions to generate scenarios inside a given map
Scenario:
        list of agents starting positions -> [(x0,y0), (x1,y1), ...]
        list of tasks -> [(task1), (task2), ...]
    Task:
        pickup and delivery positions, tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))

This file does not provide methods to feed tasks to the agents
Tasks are provided in the scenario as a list, how those are passed into the problem is managed externally
It is possible to add tasks into a scenario after its creation

Tasks locations can coincide or not with agents starting position or with other tasks locations
Both of these options are controllable with a parameter
Default -> tasks locations can't coincide with non-task endpoints in order to generate well-formed MAPD instances
"""

from random import sample

import numpy as np


def create_scenario(config, input_map):
    """
    Create a scenario
    :param config: Namespace of dataset configurations
    :param input_map: np.ndarray, shape:H*W, matrix of 0 and 1
    :return: start_pos_list -> list of agent starting positions, [(x,y), ...]
             parking_spot_list -> list of agent parking spots, [(x,y), ...]
             task_list -> [(task1), (task2), ...]
                          task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    """
    # get starting positions
    start_pos_list = create_starting_pos(input_map=input_map,
                                         agent_num=config.agent_number,
                                         mode=config.start_position_mode,
                                         fixed_pos_list=config.fixed_position_list)

    # non task endpoints list
    parking_spot_list = []  # should not contain any agent starting position
    non_task_ep_list = start_pos_list + parking_spot_list

    # get task list
    task_list = []
    for _ in range(config.task_number):
        task_list.append(create_task(input_map=input_map,
                                     mode=config.task_creation_mode,
                                     non_task_ep_list=non_task_ep_list,
                                     task_list=task_list))  # list of tasks, passed recursively

    return start_pos_list, parking_spot_list, task_list


def create_starting_pos(input_map, agent_num, mode='random', fixed_pos_list=None):
    """
    Get starting position, one for each agent, and collect them
    :param input_map: np.ndarray, shape:H*W, matrix of 0 and 1
    :param agent_num: int, number of agents
    :param mode:
                'random': randomly generates starting positions amongst free cell
                          guarantees that starting positions do not coincide with obstacles
                'fixed': choose amongst predefined, passed set of positions
                         no guarantees that fixed spots won't overwrite obstacles positions
    :param fixed_pos_list: list of fixed starting positions to choose from -> [(x,y), ...]
                           only used when 'start_pos_mode' = 'fixed'
    :return: start_pos: list of starting positions of agents -> [(x,y), ...]
    :raise: ValueError if not enough starting positions for all the agents
    """

    # fixed generation
    if mode == 'fixed':
        if agent_num > len(fixed_pos_list):
            raise ValueError('Not enough starting positions for all the agents')
        # select randomly from fixed pos list
        return sample(population=fixed_pos_list, k=agent_num)    # list of tuples

    # defaults to random generation
    else:
        # get positions of free cells
        where_res = np.nonzero(input_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))
        if agent_num > len(free_cell_list):
            raise ValueError('Not enough starting positions for all the agents')
        # select randomly amongst free cell
        return sample(population=free_cell_list, k=agent_num)  # list of tuples


def create_task(input_map, mode='avoid_non_task_rep',
                non_task_ep_list=None, task_list=None):
    """
    Return a task for the given map and starting positions
    Whether tasks can coincide with starting locations or with other tasks is controlled by 'mode'
    :param input_map: np.ndarray, shape:H*W, matrix of 0 and 1
    :param mode:
                'free'
                    no restriction on new task position (obviously avoiding obstacles)
                'avoid_non_task_rep'
                    avoid placing new task on non-task endpoints
                'avoid_task_rep'
                    avoid placing new task on another task position
                'avoid_all'
                    both 'avoid_non_task_rep' and 'avoid_task_rep'
    :param non_task_ep_list: list of non task endpoints -> [(x,y), ...]
                             if 'mode' = 'avoid_non_task_rep' or 'avoid_all' --> 'non_task_ep_list' != None
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
                      if 'mode' = 'avoid_task_rep' or 'avoid_all' --> 'task_list' != None
    :return: task: tuple ((x,y),(x,y)) -> ((pickup_pos),(delivery_pos))
    :raise: ValueError when non_task_ep_list or task_list are required but == None
    """
    # filters out obstacles coords
    where_res = np.nonzero(input_map == 0)
    free_cell_pool = set(zip(where_res[0], where_res[1]))

    # no tasks on other task positions
    if mode == 'avoid_task_rep' or mode == 'avoid_all':
        if not task_list:
            raise ValueError('Task list is required with the current mode')
        free_cell_pool = free_cell_pool - {loc
                                           for task in task_list
                                           for loc in task}

    # if wrong 'mode' input, defaults to 'avoid_non_task_rep'
    # no tasks on non-task endpoint positions
    if mode != 'free' and mode != 'avoid_task_rep':
        if not non_task_ep_list:
            raise ValueError('Non-task endpoint list is required with the current mode')
        free_cell_pool = free_cell_pool - set(non_task_ep_list)

    # get the task
    return tuple(sample(population=list(free_cell_pool), k=2))
