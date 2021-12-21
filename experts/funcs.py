"""
Utility functions for experts algorithms
"""

from scipy.spatial import distance
import numpy as np
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
    heuristic = [distance.cityblock(np.array((row, col)), np.array(goal))
                 for row in range(input_map.shape[0])
                 for col in range(input_map.shape[1])]

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


def has_valid_expansion(coord, input_map, closed_list,
                        token=None, timestep=None):
    """
    Check if is possible for a* to expand the new cell
    1) Check if the cell is inside map boundaries
    2) Check if the cell has already been expanded
    3) Check if the cell has an obstacle inside
    4) Check token for avoiding conflicts with other agents
    :param coord: (x, y), int tuple of current position
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param closed_list: implemented as matrix with shape = input_map.shape
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                  x, y -> cartesian coords, t -> timestep, starting from t=0
    :param timestep: int, current timestep, used positionally
                     e.g. timestep = 2 -> looking for tuple (x,y,t) at depth 2 in the path
                     YES: path[2]
                     NO: path[i] if path[i].t == 2
    :return: True if all 4 checks are passed, False else
    """
    x = coord[0]
    y = coord[1]

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
    if not token:
        return True

    # check 4), check that at token[timestamp] there are no conflicting cells
    # when called by token passing, all paths in the token are from different agents
    # if you have a path in the token, you don't need to compute a path and call a*
    moves_curr_ts = [path_[timestep] for path_ in token.values()
                     if len(path_) > timestep]     # [(x1_t, y1_t, t), (x2_t, y2_t, t), ..., ]
    # if attempted move not conflicting, return True
    return (x, y, timestep) not in moves_curr_ts


def preprocess_heuristics(task_list, input_map):
    """
    Since cost-minimal paths need to be found only to endpoints, the path costs from all locations to all endpoints
    are computed in a preprocessing phase
    Manhattan Heuristic is used to estimate path costs
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param input_map: np.ndarray, type=int, size:H*W, matrix of 0 and 1
    :return: heuristic_collection
             dict -> {(task) : (pickup_h, delivery_h)}
             with pickup_h, delivery_h -> np.ndarray, type=int, shape=input_map.shape,
                heuristic matrices with goal = pickup_pos and delivery_pos, respectively
    """
    heuristic_coll = {}
    # loop over each task
    for task in task_list:
        pickup_pos = task[0]
        delivery_pos = task[1]
        # get heuristic w.r.t. pickup and delivery positions
        pickup_h = compute_manhattan_heuristic(input_map=input_map, goal=pickup_pos)
        delivery_h = compute_manhattan_heuristic(input_map=input_map, goal=delivery_pos)
        # add it to the collection
        heuristic_coll[task] = (pickup_h, delivery_h)

    return heuristic_coll


def __validate_path(token, path):
    """
    Agent i can stop at (x,y) at time t_i iff no other agent j != i moves to (x,y) with time t_j <= t_i
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                  x, y -> cartesian coords, t -> timestep, starting from t=0
    :param path: [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
                 path the agent is trying to follow
    :return: True if valid, False if not
    """
    # [ (x,y,t), ... ], list of steps of all the agents in the token
    step_list = [s for p in token.values() for s in p]

    stop = path[-1]     # last step in the path
    # if (x_s, y_s) in step_list, verify that it moves to that cell after the other agent
    match = [stop[2]-step[2]
             for step in step_list
             if stop[:-1] == step[:-1]]
    # match -> [ t1, t2, ...] where t_i = stop_t_i - other_path_t_i
    # so if delta_t in match is <= 0, the agent will be blocking another
    # return True only if all delta_t > 0
    return np.all([delta_t > 0 for delta_t in match])


def find_resting_pos(start, task_list, token, input_map):
    """
    Pick the nearest endpoint s.t. delivery and pickup locations of all tasks are different from the chosen endpoint,
    no path of other agents in the token ends in the chosen endpoint
    and does not collide with the paths of other agents stored in the token
    :param start: (x, y), tuple of int with start cartesian coordinates
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                  x, y -> cartesian coords, t -> timestep, starting from t=0
    :param input_map: np.ndarray, type=int, size:H*W, matrix of 0 and 1
    :return: minimal cost path to the chosen endpoint
             [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
    """
    # get distance of other cells w.r.t. starting position
    heuristic = compute_manhattan_heuristic(input_map=input_map, goal=start)

    # copy to avoid modifications
    copy_map = input_map.copy()
    # loop over all tasks and set = 1 pickup and delivery coords
    for task in task_list:
        copy_map[task[0]] = 1        # pickup location, task[0], (x_p, y_p)
        copy_map[task[1]] = 1        # delivery location, task[1], (x_d, y_d)
    # loop over all paths in token and set = 1 their endpoints
    for path in token.values():
        end_loc = path[-1][:-1]      # get endpoint of path, only (x,y), excluding timestep
        copy_map[end_loc] = 1

    # discard not available cells, working with 1D array
    flat_map = copy_map.flatten()
    free_idx_list = np.nonzero(flat_map == 0)[0]   # [0] since it returns the list 'wrapped' by a tuple

    # get indexes that would sort flattened heuristic
    # if argmin(heuristic) = 3 -> order[0] = 3
    # if second lowest value of heuristic is in position 5 -> order[1] = 5
    # if argmax(heuristic) = n -> order[heuristic.size-1] = n
    # e.g. -> heuristic = [1, 4, 0, 8]
    #         order     = [2, 0, 1, 3]
    order = np.argsort(heuristic.flatten())

    # remove lowest since it's the agent start position
    order = np.delete(order, 0)

    ret_path = []
    # check cells in ascending order of heuristics (low first)
    while order.size > 0 and not ret_path:
        # 'pop' index of argmin
        idx = order[0]
        order = np.delete(order, 0)
        # check if the cell is free
        if idx in free_idx_list:
            target = np.unravel_index(idx, input_map.shape)     # get 2D coordinates
            try:
                ret_path = a_star(input_map=input_map, start=start, goal=target, token=token)

                '''
                found path is already collision free
                however, the code still allows for an agent to rest in the path of another agent, if the first reaches
                its endpoint before the second moves in that square
                we want to remove this, as can cause collisions
                agent i can move to (x,y) at time t_i iff no other agent j != i moves to (x,y) with time t_j <= t_i
                '''
                # if valid, return the path
                if __validate_path(token, ret_path):
                    return ret_path

            except ValueError:
                ret_path = []

    # no path found, stay in place
    return [(start[0], start[1], 0)]
