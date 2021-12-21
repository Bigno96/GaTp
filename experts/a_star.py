"""
A* search and supporting functions
Base algorithm is modified for being executed during the Token Passing algorithm

A* searches in a state space whose states are pairs of locations and timesteps.
A directed edge exists from state (l; t) to state (l_0; t + 1) iff l = l_0 or (l; l_0) belongs to E.

State (l; t) is removed from the state space iff a_i being in location l at timestep t results in it colliding
with other agents a_j that move along their paths in the token.
Similarly, the edge from state (l; t) to state (l_0; t + 1) is removed from the state space iff
a_i moving from location l to location l_0 at timestep t results in it colliding
with other agents a_j that move along their paths in the token.

The following implementation is based on:
    - A* implementation by Andrew Dahdouh,
        Copyright (c) 2017, Andrew Dahdouh.
        All rights reserved.
    - Token Passing pseudo-code as described in
        Ma, H., Li, J., Kumar, T. K., & Koenig, S. (2017).
        Lifelong multi-agent path finding for online pickup and delivery tasks.
        arXiv preprint arXiv:1705.10868.
"""

import logging
from types import SimpleNamespace

import numpy as np
from scipy.spatial import distance

from create_dataset.map_creator import MapCreator

# moves dictionary
DELTA = [(-1, 0),  # go up
         (0, -1),  # go left
         (1, 0),  # go down
         (0, 1)]  # go right


def __compute_heuristic(input_map, goal):
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


def __is_goal(coord, goal):
    """
    Return True if the current coordinates are the goal, False otherwise
    :param coord: (x, y), int tuple of current position
    :param goal: (x, y), int tuple of goal position
    :return: boolean
    """
    # curr_x = goal_x & curr_y = goal_y
    return coord[0] == goal[0] and coord[1] == goal[1]


def __is_valid(coord, input_map, closed_list, token=None, timestep=None):
    """
    Check if is possible to expand the new cell
    1) Check if the cell is inside map boundaries
    2) Check if the cell has already been expanded
    3) Check if the cell has an obstacle inside
    4) Check token for avoiding conflicts with other agents
    :param coord: (x, y), int tuple of current position
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param closed_list: implemented as matrix with shape = input_map.shape
    :param token: summary of other agents planned paths
                  Namespace -> agent_id : path
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

    # check 4), check that at token[timestamp] there are no conflicting cells
    # when called by token passing, all paths in the token are from different agents
    # if you have a path in the token, you don't need to compute a path and call a*
    moves_curr_ts = [path_[timestep] for path_ in token.__dict__.values()
                     if len(path_) > timestep]     # [(x1_t, y1_t, t), (x2_t, y2_t, t), ..., ]
    # if attempted move not conflicting, return True
    return (x, y, timestep) not in moves_curr_ts


def a_star(input_map, start, goal, token=None):
    """
    A* Planner method
    Finds a plan from a starting node to a goal node if one exists
    Custom collision avoidance:
        not going into a cell if another agent is already scheduled to go there at that timestamp
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param start: (x, y), tuple of int with start cartesian coordinates
    :param goal: (x, y), tuple of int with goal cartesian coordinates
    :param token: summary of other agents planned paths
                  Namespace -> agent_id : path
                  with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                  x, y -> cartesian coords, t -> timestep, starting from t=0
            Default: None, defaults to classic A*
    :return: path_found: [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
             path_length: int
    :raise ValueError if no path are found
    """
    # pre-compute heuristic
    heuristic = __compute_heuristic(input_map=input_map,
                                    goal=goal)

    '''
    Supporting data structures
    '''
    # closed list, implemented as matrix with shape = input_map.shape
    closed_list = np.zeros(input_map.shape, dtype=int)
    # delta_tracker, used to track moves for reconstructing return path
    delta_tracker = np.full(input_map.shape, fill_value=-1, dtype=int)

    '''
    Initialization
    '''
    x = start[0]
    y = start[1]

    cost = 1  # cost of each step
    g = 0  # cost of the path to the current cell
    f = g + heuristic[x][y]
    t = 0  # timestep

    open_list = [(f, g, x, y, t)]  # fringe
    closed_list[x][y] = 1  # visit the starting cell

    '''
    Main execution loop
    '''
    # while open list is not empty
    while open_list:
        open_list.sort()  # ascending order of f
        q = open_list.pop(0)  # pop first, min(f)
        g = q[1]
        x = q[2]
        y = q[3]
        timestep = q[4]

        # if goal is reached
        if __is_goal(coord=(x, y), goal=goal):
            full_path = []
            # loop back until start is reached
            while x != start[0] or y != start[1]:
                previous_x = x - DELTA[delta_tracker[x][y]][0]
                previous_y = y - DELTA[delta_tracker[x][y]][1]
                full_path.append((x, y, timestep))  # (x_t, y_t, t)
                # trace back
                x = previous_x
                y = previous_y
                timestep -= 1

            full_path.append((start[0], start[1], 0))
            full_path.reverse()

            return full_path, len(full_path)

        else:
            # keep track of the timestep when the node was popped
            timestep += 1
            # for each possible move
            for idx, move in enumerate(DELTA):
                x_next = x + move[0]
                y_next = y + move[1]
                # if the point is valid for the expansion
                if __is_valid(coord=(x_next, y_next),
                              input_map=input_map, closed_list=closed_list,
                              token=token, timestep=timestep):
                    # update values and append to the fringe
                    g_next = g + cost
                    f = g_next + heuristic[x_next][y_next]
                    open_list.append((f, g_next, x_next, y_next, timestep))
                    closed_list[x_next][y_next] = 1                     # node has been visited
                    delta_tracker[x_next][y_next] = idx                 # keep track of the move

    raise ValueError('No path found')


# test a_star
# mind that start or end position, since hardcoded, might end up being in an obstacle position, therefore unreachable
if __name__ == '__main__':
    __spec__ = None
    map_creator = MapCreator(map_size=(10, 10),
                             map_density=0.2)
    random_grid_map = map_creator.create_random_grid_map()
    print(random_grid_map)

    tok = SimpleNamespace()
    tok.one = [(2, 2, 0), (1, 2, 1), (1, 3, 2)]
    tok.two = [(3, 1, 0), (2, 1, 1), (1, 1, 2)]

    try:
        path, length = a_star(input_map=random_grid_map,
                              start=(1, 1),
                              goal=(8, 8),
                              token=tok)
        print(path)

    except ValueError as err:
        logging.getLogger().warning(err)
