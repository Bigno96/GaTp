"""
A* search
Base algorithm is modified for being executed during the Token Passing algorithm

A* searches in a state space whose states are pairs of locations and timesteps.
A directed edge exists from state (l; t) to state (l_0; t+1) iff l = l_0 or (l; l_0) belongs to E.

State (l; t) is removed from the state space iff a_i being in location l at timestep t results in it colliding
with other agents a_j that move along their paths in the token.
Similarly, the edge from state (l; t) to state (l_0; t+1) is removed from the state space iff
a_i moving from location l to location l_0 at timestep t results in it colliding
with other agents a_j that move along their paths in the token.

The following implementation is based on:
    - A* implementation by Andrew Dahdouh,
        Copyright (c) 2017, Andrew Dahdouh.
        All rights reserved.
    - Token Passing pseudocode as described in
        Ma, H., Li, J., Kumar, T. K., & Koenig, S. (2017).
        Lifelong multiagent path finding for online pickup and delivery tasks.
        arXiv preprint arXiv:1705.10868.
"""

import logging

import numpy as np
from experts.funcs import compute_manhattan_heuristic, is_goal, has_valid_expansion

from create_dataset.map_creator import MapCreator

# moves dictionary
DELTA = [(-1, 0),  # go up
         (0, -1),  # go left
         (1, 0),  # go down
         (0, 1)]  # go right


def a_star(input_map, start, goal,
           token=None, heuristic=None):
    """
    A* Planner method
    Finds a plan from a starting node to a goal node if one exists
    Custom collision avoidance:
        not going into a cell if another agent is already scheduled to go there at that timestamp
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param start: (x, y), tuple of int with start cartesian coordinates
    :param goal: (x, y), tuple of int with goal cartesian coordinates
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                  x, y -> cartesian coords, t -> timestep, starting from t=0
            Default: None, defaults to classic A*
    :param heuristic: np.ndarray, type=int, heuristic.shape = input_map shape
                      given heuristic matrix
            Default: None, computes manhattan heuristic
    :return: path_found: [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
             path_length: int
    :raise ValueError if no path are found
    """
    # pre-compute heuristic if none
    if not heuristic:
        heuristic = compute_manhattan_heuristic(input_map=input_map,
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
    f = g + heuristic[(x, y)]
    t = 0  # timestep

    open_list = [(f, g, x, y, t)]  # fringe
    closed_list[(x, y)] = 1  # visit the starting cell

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

        coord = (x, y)
        # if goal is reached
        if is_goal(coord=coord, goal=goal):
            full_path = []
            # loop back until start is reached
            while x != start[0] or y != start[1]:
                previous_x = x - DELTA[delta_tracker[coord]][0]
                previous_y = y - DELTA[delta_tracker[coord]][1]
                full_path.append((x, y, timestep))  # (x_t, y_t, t)
                # trace back
                x = previous_x
                y = previous_y
                coord = (x, y)
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
                next_c = (x_next, y_next)
                # if the point is valid for the expansion
                if has_valid_expansion(coord=next_c,
                                       input_map=input_map, closed_list=closed_list,
                                       token=token, timestep=timestep):
                    # update values and append to the fringe
                    g_next = g + cost
                    f = g_next + heuristic[next_c]
                    open_list.append((f, g_next, x_next, y_next, timestep))
                    closed_list[next_c] = 1                     # node has been visited
                    delta_tracker[next_c] = idx                 # keep track of the move

    raise ValueError('No path found')


# test a_star
# mind that start or end position, since hardcoded, might end up being in an obstacle position, therefore unreachable
if __name__ == '__main__':
    __spec__ = None
    map_creator = MapCreator(map_size=(10, 10),
                             map_density=0.2)
    random_grid_map = map_creator.create_random_grid_map()
    print(random_grid_map)

    tok = {0: [(2, 2, 0), (1, 2, 1), (1, 3, 2)],
           1: [(3, 3, 0), (3, 4, 1), (4, 4, 2)]}

    try:
        path, length = a_star(input_map=random_grid_map,
                              start=(1, 1),
                              goal=(8, 8),
                              token=tok)
        print(path)

    except ValueError as err:
        logging.getLogger().warning(err)
