"""
Utility functions for experts algorithms
"""

import numpy as np
from scipy.spatial import distance

from a_star import a_star


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
    heuristic = [distance.cityblock((row, col), goal)
                 for row in range(input_map.shape[0])
                 for col in range(input_map.shape[1])
                 ]

    return np.array(heuristic, dtype=int).reshape(input_map.shape)


def is_goal(coord, goal):
    """
    Return True if the current coordinates are the goal, False otherwise
    :param coord: (x, y), int tuple of current position
    :param goal: (x, y), int tuple of goal position
    :return: boolean
    """
    # curr_x = goal_x & curr_y = goal_y
    return coord[0] == goal[0] and coord[1] == goal[1]


def has_valid_expansion(next_pos, curr_pos, input_map, closed_list,
                        token=None, timestep=None):
    """
    Check if is possible for a* to expand the new cell
    1) Check if the cell is inside map boundaries
    2) Check if the cell has already been expanded
    3) Check if the cell has an obstacle inside
    4) Check token for avoiding conflicts with other agents
    :param next_pos: (x, y), int tuple of new position
    :param curr_pos: (x, y), int tuple of curr position
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param closed_list: implemented as matrix with shape = input_map.shape
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                  x, y -> cartesian coords, t -> timestep
    :param timestep: int, timestep of new expansion, used positionally
                     e.g. timestep = 2 -> looking for tuple (x,y,t) at depth 2 in the path
                     YES: path[2]
                     NO: path[i] if path[i].t == 2
    :return: True if all 4 checks are passed, False else
    """
    x = next_pos[0]
    y = next_pos[1]

    # check 1), curr_x < shape_x & curr_x >= 0 & curr_y >= 0 & curr_y < shape_y
    if x >= input_map.shape[0] or x < 0 or y >= input_map.shape[1] or y < 0:
        return False

    # check 2), the current node has not been expanded
    if closed_list[x][y] == 1:
        return False

    # check 3), the current cell is not an obstacle (not 1)
    if input_map[x][y] == 1:
        return False

    # classic A*
    if not token or not timestep:
        return True

    # check 4), check that at token[timestamp] there are no conflicting cells
    # when called by token passing, all paths in the token are from different agents
    # timestep always > 0

    # no swap constraint
    bad_moves_list = [(x_s, y_s, timestep)
                      for path_ in token.values()
                      for x_s, y_s, _ in path_[timestep-1]     # avoid going into their current position next move
                      if len(path_) > timestep
                      and path_[timestep][:-1] == curr_pos      # if that agent is going into my current position
                      ]

    # avoid node collision
    bad_moves_list.extend([path_[timestep]          # [(x1_t, y1_t, t), (x2_t, y2_t, t), ..., ]
                           for path_ in token.values()
                           if len(path_) > timestep
                           ])
    # add also coordinates of agent resting on a spot
    bad_moves_list.extend([(x_s, y_s, timestep)
                           for path_ in token.values
                           for x_s, y_s, t_s in path_
                           # timestep = 0 and only 1 step -> agent is resting
                           if len(path_) == 1 and t_s == 0
                           ])

    # if attempted move not conflicting, return True
    return (x, y, timestep) not in set(bad_moves_list)


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


def find_resting_pos(start, input_map, token, h_coll,
                     task_list, non_task_ep_list):
    """
    Pick the nearest endpoint s.t. delivery locations of all tasks are different from the chosen endpoint,
    no path of other agents in the token ends in the chosen endpoint
    and does not collide with the paths of other agents stored in the token
    :param start: (x, y), tuple of int with start cartesian coordinates
    :param input_map: np.ndarray, type=int, size:H*W, matrix of 0 and 1
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                  x, y -> cartesian coords, t -> timestep
    :param h_coll: dict -> {endpoint : h_map}
                   endpoint: tuple (x,y) of int coordinates
                   h_map: np.ndarray, type=int, shape=input_map.shape, heuristic matrices with goal = endpoint
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param non_task_ep_list: list of endpoints not belonging to a task -> [(ep1), (ep2), ...]
                             endpoint: tuple (x,y) of int coordinates
    :return: minimal cost path to the chosen endpoint
             [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
    """
    # task related endpoints, excluding all delivery locations in task_list
    # -> get only pickup locations
    ep_list = [pickup
               for pickup, _ in task_list]
    # add non task related endpoints
    ep_list.extend(non_task_ep_list)

    # get list of all endpoints in the token (cutting off timesteps)
    token_ep_list = [path[-1][:-1]
                     for path in token.values()]
    # remove an endpoint if it's an endpoint also for another agent's path in the token
    ep_list = list(set(ep_list) - set(token_ep_list))

    # get heuristic for each endpoint from start position
    h_ep_list = [h_coll[ep][start]
                 for ep in ep_list]
    # list of idx, corresponding to an ordering of ep_list
    # if h_ordering_idx[i] = 0 -> the nearest endpoint is ep_list[i]
    # if h_ordering_idx[j] = 1 -> the second-nearest endpoint is ep_list[j]
    h_ordering_idx = np.argsort(h_ep_list)

    # while there are still endpoints to try
    while h_ordering_idx:
        # pop argmin -> obtain 'i', index of the nearest endpoint
        best_ep_idx = np.argmin(h_ordering_idx)
        np.delete(h_ordering_idx, best_ep_idx)
        # get best endpoint
        best_ep = ep_list[best_ep_idx]

        try:
            # collision free path, if endpoint is reachable
            path, _ = a_star(input_map=input_map, start=start, goal=best_ep,
                             token=token, heuristic=h_ep_list[best_ep_idx])
            return path

        except ValueError:
            pass    # keep going and try another endpoint

    # no endpoint was reachable -> stay in place
    # this happens due to MAPD instance not being well-formed
    return [(start[0], start[1], 0)]
