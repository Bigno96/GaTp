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
    - Token Passing pseudocode as described in
        Ma, H., Li, J., Kumar, T. K., & Koenig, S. (2017).
        Lifelong multiagent path finding for online pickup and delivery tasks.
        arXiv preprint arXiv:1705.10868.
"""

import heapq
from collections import deque
from math import hypot

import numpy as np

from utils.expert_utils import compute_manhattan_heuristic, get_next_node_list, NoPathError


def a_star(input_map, start, goal, include_start_node,
           token=None, h_map=None, starting_t=0):
    """
    A* Planner method
    Finds a plan from a starting node to a goal node if one exists
    Custom collision avoidance:
        not going into a cell if another agent is already scheduled to go there at that timestamp
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param start: (x, y), tuple of int with start cartesian coordinates
    :param goal: (x, y), tuple of int with goal cartesian coordinates
    :param include_start_node: bool, whether to include starting position (visited at time = starting_t)
                                into the returned plan or not
                                if False, start is considered visited at time = starting_t - 1
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
    :raise NoPathError if no paths are found
    """
    # fist check, if start == goal
    if start == goal:
        return deque([(start[0], start[1], 0)]), 1

    # pre-compute heuristic if none
    if not isinstance(h_map, np.ndarray):
        h_map = compute_manhattan_heuristic(input_map=input_map, goal=goal)

    '''
    Supporting data structures
    '''
    # closed list, implemented as matrix with shape = input_map.shape
    closed_list = set()
    # back_tracker, used to track moves for reconstructing return path
    back_tracker = {}

    '''
    Initialization
    '''
    if not include_start_node:      # no starting node
        starting_t -= 1
    start_node = (start[0], start[1], starting_t)
    # max path length
    max_depth = starting_t + int(hypot(input_map.shape[0], input_map.shape[1]) * 1.5)

    g = 0  # cost of the path to the current cell
    f = g + h_map[start]
    cost = 1  # cost of each step

    open_list = [(f, g, start_node)]  # fringe
    heapq.heapify(open_list)    # priority queue in ascending order of f
    closed_list.add(start_node)  # visit the starting node

    '''
    Main execution loop
    '''
    # while open list is not empty
    while open_list:
        _, g, curr_node = heapq.heappop(open_list)

        # if goal is reached
        if curr_node[:-1] == goal:
            path = deque()
            # loop back until start is reached and build full path
            while curr_node != start_node:
                path.appendleft(curr_node)
                curr_node = back_tracker[curr_node]
            # add start node if included
            if include_start_node:
                path.appendleft(start_node)

            return path, len(path)

        else:
            for next_node in get_next_node_list(curr_node=curr_node, max_depth=max_depth, starting_t=starting_t,
                                                input_map=input_map, closed_list=closed_list, token=token):
                # update values and append to the fringe
                closed_list.add(next_node)  # node has been visited
                back_tracker[next_node] = curr_node  # keep track of the move
                g_next = g + cost
                f = g_next + h_map[next_node[:-1]]
                heapq.heappush(open_list, (f, g_next, next_node))

    raise NoPathError('No path found')
