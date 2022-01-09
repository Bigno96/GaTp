"""
Utility functions for experts algorithms
"""

import numpy as np
from operator import sub

# delta dictionary
DELTA = [(-1, 0),  # go up
         (0, -1),  # go left
         (1, 0),  # go down
         (0, 1)]  # go right

# move dictionary
MOVE_LIST = [(-1, 0),  # go up
             (0, -1),  # go left
             (1, 0),  # go down
             (0, 1),  # go right
             (0, 0)]  # stay still


def compute_manhattan_heuristic(input_map, goal):
    """
    Create a matrix the same shape of the input map
    Calculate the cost from the goal node to every other node on the map using MANHATTAN heuristic
    Return the heuristic matrix
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param goal: (x, y), tuple of int with goal cartesian coordinates
    :return: heuristic, np.ndarray, heuristic.shape = input_map.shape
    """
    # abs(current_cell.x – goal.x) + abs(current_cell.y – goal.y)
    heuristic = [(np.abs(row - goal[0]) + np.abs(col - goal[1]))
                 for row in range(input_map.shape[0])
                 for col in range(input_map.shape[1])
                 ]

    return np.array(heuristic, dtype=int).reshape(input_map.shape)


def is_valid_expansion(child_pos, input_map, closed_list,
                       parent_pos=None, token=None, child_timestep=None):
    """
    Check if is possible for a* to expand the new cell
    1) Check if the cell is inside map boundaries
    2) Check if the cell has already been expanded
    3) Check if the cell has an obstacle inside
    4) Check token for avoiding conflicts with other agents
    :param child_pos: (x, y), int tuple of new position
    :param parent_pos: (x, y), int tuple of curr position
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param closed_list: implemented as matrix with shape = input_map.shape, np.ndarray
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...])
                  x, y -> cartesian coords, t -> timestep
    :param child_timestep: int, timestep of new expansion, used positionally
                     e.g. timestep = 2 -> looking for tuple (x,y,t) at depth 2 in the path
                     YES: path[2]
                     NO: path[i] if path[i].t == 2
    :return: True if all 4 checks are passed, False else
    """
    x, y = child_pos

    # check 1), x < shape_x & x >= 0 & y >= 0 & y < shape_y
    if x < 0 or x >= input_map.shape[0] or y < 0 or y >= input_map.shape[1]:
        return False

    # check 2), the current node has not been expanded
    if closed_list[child_pos] == 1:
        return False

    # check 3), the current cell is not an obstacle (not 1)
    if input_map[child_pos] == 1:
        return False

    # defaults to classic A*
    if not token or not child_timestep or not parent_pos:
        return True

    # check 4), check that at token[timestamp] there are no conflicting cells
    # when called by token passing, all paths in the token are from a different agent
    # child_timestep always > 0 by construction

    # no swap constraint
    bad_moves_list = [(x_s, y_s, child_timestep)
                      for path in token.values()
                      for x_s, y_s, t_s in path
                      if len(path) > child_timestep
                      and t_s == child_timestep  # avoid going into their current position next move
                      and path[child_timestep][:-1] == parent_pos  # if that agent is going into my current position
                      ]

    # avoid node collision
    bad_moves_list.extend([path[child_timestep]  # [(x1_t, y1_t, t), (x2_t, y2_t, t), ..., ]
                           for path in token.values()
                           if len(path) > child_timestep
                           ])

    # add also coordinates of agent resting on a spot
    bad_moves_list.extend([(path[-1][0], path[-1][1], child_timestep)
                           for path in token.values()
                           # agent is potentially resting there
                           if len(path) <= child_timestep 
                           ])

    # if attempted move not conflicting, return True
    return (x, y, child_timestep) not in bad_moves_list


def preprocess_heuristics(input_map, task_list, non_task_ep_list):
    """
    Since cost-minimal paths need to be found only to endpoints, the path costs from all locations to all endpoints
    are computed in a preprocessing phase
    Manhattan Heuristic is used to estimate path costs
    :param input_map: np.ndarray, type=int, size:H*W, matrix of 0 and 1
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param non_task_ep_list: list of endpoints not belonging to a task -> [(ep1), (ep2), ...]
                             endpoint: tuple (x,y) of int coordinates
    :return: heuristic_collection
             dict -> {endpoint : h_map}
             endpoint: tuple (x,y) of int coordinates
             h_map: np.ndarray, type=int, shape=input_map.shape, heuristic matrices with goal = endpoint
    """
    # task related endpoints
    ep_list = [ep
               for task in task_list
               for ep in task]
    # add non task related endpoints
    ep_list.extend(non_task_ep_list)

    # compute h_map list
    h_map_list = [compute_manhattan_heuristic(input_map=input_map, goal=ep)
                  for ep in ep_list]

    # return dictionary
    return dict(zip(iter(ep_list), iter(h_map_list)))


def transform_agent_schedule(agent_schedule):
    """
    A matrix-form notation is used to represent produced agent schedule
    This is done in order to feed the neural network of the GaTp agent
    :param agent_schedule: {agent_id : schedule}
                            with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
    :return: matrix -> 5 (5 actions, z-axis) x num_agent (x-axis) x makespan (max path length, y-axis)
    """
    num_agent = len(agent_schedule)
    # get makespan (all paths are the same length, since everyone waits standing still the ending)
    makespan = len(agent_schedule[0])-1

    # matrix -> actions (z-axis) x num_agent (x-axis) x makespan (y-axis)
    # 5 actions order: go_up, go_left, go_down, go_right, stay_still
    matrix = np.zeros(shape=(5, num_agent, makespan), dtype=np.int8)

    # iterate over all agent's schedules
    for agent, schedule in agent_schedule.items():

        # remove timesteps
        schedule = [(x, y) for (x, y, t) in schedule]
        # this will pair schedule[i] with schedule[i-1], starting from i = 1
        zip_offset = list(zip(schedule[1:], schedule))
        # get difference between each pair in zip_offset
        diff_list = [tuple(map(sub, a, b))
                     for (a, b) in zip_offset]

        # get corresponding index in moves dictionary
        diff: tuple[int, int]
        move_idx_list = [MOVE_LIST.index(diff)
                         for diff in diff_list]

        # update matrix: actions x num_agent x makespan
        #   agent -> x coord in np.array, agent number
        #   t -> y coord in np.array, timestep
        #   move_idx -> z coord in np.array, move performed by agent at timestep t
        for t, move_idx in enumerate(move_idx_list):
            matrix[(move_idx, agent, t)] = 1

    return matrix
