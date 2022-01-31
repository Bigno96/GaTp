"""
Utility files for unit testing modules
"""

import numpy as np
import random
from copy import deepcopy
from collections import deque

from create_dataset.map_creator import create_random_grid_map
from create_dataset.scenario_creator import create_task, create_starting_pos


# create a grid map, no connection property check
def create_grid_test_map(shape, density):
    size = shape[0] * shape[1]
    obstacle_count = int(size * density)

    flat_map = np.zeros(size, dtype=np.int8)  # array of zero, dim: h*w
    # get a random permutation of numbers between 0 (included) and 'cell_count' (excluded)
    p = np.random.permutation(size)
    # set to 1 the elements give by the first 'obstacle_count' indexes of the permutation
    flat_map[p[:obstacle_count]] = 1
    # reshape as matrix
    grid_map = flat_map.reshape(shape)

    return grid_map


# find a start and a goal position inside given input map
def find_start_goal(input_map, free_cell_list=None):
    if not free_cell_list:
        where_res = np.nonzero(input_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))

    return random.sample(population=free_cell_list, k=2)


# build a task list with mode = 'free'
def build_free_task_list(input_map, length):
    task_list = [create_task(input_map=input_map, mode='free')
                 for _ in range(length)]

    return task_list


# get a grid map, a free cell list of that grid map and a token
def get_grid_map_free_cell_token(shape, density, agent_num, token_path_length):
    # map creation
    grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

    # get free cell positions
    where_res = np.nonzero(grid_map == 0)
    free_cell_list = list(zip(where_res[0], where_res[1]))

    # cell pool, avoid repetition
    pool = random.sample(population=free_cell_list, k=int((token_path_length+1) * agent_num) + 1)
    start_pos_pool = pool[:agent_num]
    path_pool = pool[agent_num:-1]

    # token
    token = {}
    for i in range(agent_num-1):
        token[i] = {'pos': start_pos_pool[i],
                    'path': deque([(x, y, t)
                                   for t, (x, y)
                                   in enumerate(path_pool[int(i * token_path_length):
                                                          int((i+1) * token_path_length)])
                                   ])
                    }
    x, y = pool[-1]
    token['stands_still'] = {'pos': (x, y), 'path': [(x, y, 0)]}  # one agent stands still

    return grid_map, free_cell_list, token


# get agents starting positions over input map, non task-endpoint list and task list
def get_start_pos_non_tep_task_list(input_map, agent_num, task_num):
    # non task endpoints
    start_pos_list = create_starting_pos(input_map=input_map, agent_num=agent_num,
                                         mode='random')
    parking_spot = []
    non_task_ep_list = start_pos_list + parking_spot

    # task list
    task_list = []
    for _ in range(task_num):
        task_list.append(create_task(input_map=input_map, mode='avoid_non_task_rep',
                                     non_task_ep_list=non_task_ep_list))

    return start_pos_list, non_task_ep_list, task_list


# get token all position list, token starting position list and token endpoint list
def get_tok_posl_startl_epl(token):
    token_pos_list = [(x, y)
                      for val in token.values()
                      for x, y, t in val['path']
                      ]
    token_start_pos_list = [val['path'][0][:-1]
                            for val in token.values()]
    token_ep_list = [val['path'][-1][:-1]
                     for val in token.values()]

    return token_pos_list, token_start_pos_list, token_ep_list


# build agent schedule from token bounding the length over the action of the specified agent
def build_ag_schedule(token):
    agent_schedule = deepcopy(token)
    # make all the paths in the token the same length
    min_len = min([len(val['path'])
                   for val in token.values()
                   ])

    for ag, val in token.items():
        path = list(val['path'])
        # path is a deque
        agent_schedule[ag] = deque(path[:min_len])

    return agent_schedule
