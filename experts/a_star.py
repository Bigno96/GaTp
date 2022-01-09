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

import heapq
from collections import deque

import numpy as np

from utils.expert_utils import compute_manhattan_heuristic, is_valid_expansion, DELTA


def a_star(input_map, start, goal,
           token=None, h_map=None, starting_t=0):
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
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...])
                  x, y -> cartesian coords, t -> timestep
            Default: None, defaults to classic A*
    :param h_map: np.ndarray, type=int, heuristic.shape = input_map shape
                  given heuristic matrix of goal
           Default: None, computes manhattan heuristic
    :param starting_t: int, token timestep at which the path it's starting
    :return: path_found: deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)])
             path_length: int
    :raise ValueError if no path are found
    """
    # pre-compute heuristic if none
    if not isinstance(h_map, np.ndarray):
        h_map = compute_manhattan_heuristic(input_map=input_map, goal=goal)

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
    x, y = start
    g = 0  # cost of the path to the current cell
    f = g + h_map[(x, y)]
    t = starting_t  # timestep
    cost = 1  # cost of each step

    open_list = [(f, g, x, y, t)]  # fringe
    heapq.heapify(open_list)    # priority queue in ascending order of f
    closed_list[(x, y)] = 1  # visit the starting cell

    '''
    Main execution loop
    '''
    # while open list is not empty
    while open_list:
        _, g, x, y, t = heapq.heappop(open_list)

        curr_c = (x, y)
        # if goal is reached
        if curr_c == goal:
            path = deque()
            # loop back until start is reached and build full path
            while curr_c != start:
                previous_x = x - DELTA[delta_tracker[curr_c]][0]
                previous_y = y - DELTA[delta_tracker[curr_c]][1]
                path.appendleft((x, y, t))  # (x_t, y_t, t)
                # trace back
                x = previous_x
                y = previous_y
                curr_c = (x, y)
                t -= 1
            # add start
            path.appendleft((start[0], start[1], t))

            return path, len(path)

        else:
            # keep track of the timestep when the node was popped
            t += 1
            # for each possible move
            for idx, move in enumerate(DELTA):
                x_next = x + move[0]
                y_next = y + move[1]
                next_c = (x_next, y_next)
                # if the point is valid for the expansion
                if is_valid_expansion(child_pos=next_c, input_map=input_map, closed_list=closed_list,
                                      parent_pos=curr_c, token=token, child_timestep=t):
                    # update values and append to the fringe
                    closed_list[next_c] = 1  # node has been visited
                    delta_tracker[next_c] = idx  # keep track of the move
                    g_next = g + cost
                    f = g_next + h_map[next_c]
                    heapq.heappush(open_list, (f, g_next, x_next, y_next, t))

    raise ValueError('No path found')
