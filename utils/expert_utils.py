"""
Utility functions for experts algorithms
"""

import numpy as np

from typing import Any, Optional
from collections import deque


# delta dictionary
MOVE_LIST = [(-1, 0),  # go up
             (0, -1),  # go left
             (1, 0),  # go down
             (0, 1),  # go right
             (0, 0)]  # stay still

# list of neighbours
NEIGHBOUR_LIST = MOVE_LIST[:-1]


# custom error for Path not Found
class NoPathError(Exception):
    pass


class StopToken:
    """
    Used to stop execution of experts instance that hangs or takes too long
    """
    def __init__(self):
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True


def is_valid_expansion(next_node: tuple[int, int, int],
                       input_map: np.array,
                       closed_list: set[tuple[int, int, int]]
                       ) -> bool:
    """
    Check if is possible for A* to expand the new cell
    1) Check if the cell is inside map boundaries
    2) Check if the cell has already been expanded
    3) Check if the cell has an obstacle inside
    :param next_node: (x, y, t), new node
    :param input_map: shape=(H, W),matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param closed_list: implemented as a set of nodes
    :return: True if all 3 checks are passed, False else
    """
    x, y = next_node[:-1]

    # check 1), x < shape_x & x >= 0 & y >= 0 & y < shape_y
    if x < 0 or x >= input_map.shape[0] or y < 0 or y >= input_map.shape[1]:
        return False

    # check 2), the current node has not been expanded
    if next_node in closed_list:
        return False

    # check 3), the current cell is not an obstacle (not == 1)
    if input_map[(x, y)] == 1:
        return False

    return True


def check_token_conflicts(token: Optional[dict[int, dict[str, tuple[int, int] or deque[tuple[int, int, int]]]]],
                          next_node: Optional[tuple[int, int, int]],
                          curr_node: Optional[tuple[int, int, int]],
                          starting_t: int = 0
                          ) -> bool:
    """
    Check that at new_pos there are no conflicts in token
    Assumes its own path is not in the token
    :param token: summary of other agents planned paths
                  dict -> {agent_name : {'pos': (x,y), 'path': path}}
                  with pos = current agent pos
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...]), future steps
                  x, y -> cartesian coords, t -> timestep
    :param curr_node: (x, y, t), current parent node
    :param next_node: (x, y, t), new node being expanded
    :param starting_t: starting timestep of the search
                       (default: 0)
    :return: True if no conflicts are found, False else
    """
    # if something is not specified, defaults to True
    if not token or not next_node or not curr_node:
        return True

    curr_timestep = curr_node[-1]
    new_timestep = next_node[-1]
    curr_pos = curr_node[:-1]
    next_pos = next_node[:-1]

    # when called, all paths in the token are from a different agent
    # construct list of bad moves
    bad_moves_list = [(x_s, y_s)
                      for val in token.values()
                      for x_s, y_s, t_s in val['path']
                      # no swap constraint
                      if (t_s == curr_timestep  # avoid going into their current position next move
                          # if that agent is going into my current position
                          and curr_pos in [(x_p, y_p)
                                           for x_p, y_p, t_p in val['path']
                                           if t_p == new_timestep]
                          )
                      # avoid node collision
                      or t_s == new_timestep
                      ]

    # if another agent a_i is moving into curr_pos at timestep == starting_t
    # bad_moves_list still allows next_pos == loc(a_i) -> swap conflict
    if curr_timestep == starting_t:
        # only possible at first depth expansions
        bad_moves_list.extend([val['pos']   # don't go into his current position
                               for val in token.values()
                               # if that agent is going into my current position
                               if val['path'][0][:-1] == curr_pos
                               ])

    # if attempted move not conflicting, return True
    return next_pos not in bad_moves_list


def get_next_node_list(curr_node: tuple[int, int, int],
                       max_depth: int,
                       starting_t: int,
                       input_map: np.array,
                       closed_list: set[tuple[int, int, int]],
                       token: dict[int, dict[str, tuple[int, int] or deque[tuple[int, int, int]]]]
                       ) -> list[tuple[int, int, int]]:
    """
    Get list of nodes available for A* expansion
    :param curr_node: (x, y, t), current parent node
    :param max_depth: maximum path depth
    :param starting_t: starting timestep for the search
    :param input_map: shape=(H, W), matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param closed_list: implemented as a set of nodes
    :param token: summary of other agents planned paths
                  dict -> {agent_name : {'pos': (x,y), ''path': path}}
                  with pos = current agent pos
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...]), future steps
                  x, y -> cartesian coords, t -> timestep
    :return: list of 'next_nodes' available for expansion
    :raise ValueError if path is longer than max_depth
    """

    # get position and timestep
    x, y, t = curr_node

    # avoid infinite looping
    if t > max_depth:
        raise NoPathError('No path found')

    # for each possible move
    next_node_list = [(x+move[0], y+move[1], t+1)
                      for move in MOVE_LIST]

    # filter out invalid node for the expansion
    return [next_node
            for next_node in next_node_list
            if is_valid_expansion(next_node=next_node, input_map=input_map, closed_list=closed_list)
            and check_token_conflicts(token=token, next_node=next_node, curr_node=curr_node, starting_t=starting_t)
            ]


def compute_manhattan_heuristic(input_map: np.array,
                                goal: tuple[int, int]
                                ) -> np.array:
    """
    Create a matrix the same shape of the input map
    Calculate the cost from the goal node to every other node on the map using MANHATTAN heuristic
    Return the heuristic matrix
    :param input_map: shape=(H, W), matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param goal: (x, y), goal cartesian coordinates
    :return: computed heuristic, shape=input_map.shape
    """
    # abs(current_cell.x – goal.x) + abs(current_cell.y – goal.y)
    heuristic = [(abs(row - goal[0]) + abs(col - goal[1]))
                 for row in range(input_map.shape[0])
                 for col in range(input_map.shape[1])
                 ]
    return np.array(heuristic, dtype=int).reshape(input_map.shape)


def preprocess_heuristics(input_map: np.ndarray,
                          task_list: list[tuple[tuple[int, int], tuple[int, int]]],
                          non_task_ep_list: list[tuple[int, int]]
                          ) -> dict[tuple[int, int], np.array]:
    """
    Since cost-minimal paths need to be found only to endpoints, the path costs from all locations to all endpoints
    are computed in a preprocessing phase
    Manhattan Heuristic is used to estimate path costs
    :param input_map: shape=(H, W), matrix of 0 and 1
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param non_task_ep_list: list of endpoints not belonging to a task -> [(ep1), (ep2), ...]
                             endpoint: (x,y), endpoint coordinates
    :return: heuristic collection
             dict -> {endpoint : h_map}
             endpoint: (x,y), endpoint coordinates
             h_map: shape=input_map.shape, heuristic matrices with goal = endpoint
    """
    # task related endpoints
    ep_list = [ep
               for task in task_list
               for ep in task]
    # add non task related endpoints
    ep_list.extend(non_task_ep_list)
    # tuple conversion (for hashing safeness)
    ep_list = [tuple(ep) for ep in ep_list]

    # compute h_map list
    h_map_list = [compute_manhattan_heuristic(input_map=input_map, goal=ep)
                  for ep in ep_list]

    # return dictionary
    return dict(zip(iter(ep_list), iter(h_map_list)))


def free_cell_heuristic(target: tuple[int, int],
                        input_map: np.array,
                        token: dict[int, dict[str, tuple[int, int] or deque[tuple[int, int, int]]]],
                        target_timestep: int
                        ) -> int:
    """
    Get how many cells are free around target at specified timestep
    :param target: (x,y), cartesian coords of the target
    :param input_map: matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param token: summary of other agents planned paths
                  dict -> {agent_name : {'pos': (x,y), ''path': path}}
                  with pos = current agent pos
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...]), future steps
                  x, y -> cartesian coords, t -> timestep
    :param target_timestep: global timestep of the execution
    :return: value of free cells around
    """
    # get agent adjacent cells
    target_neighbours = [(target[0]+move[0], target[1]+move[1])
                         for move in NEIGHBOUR_LIST]    # exclude standing still to find neighbours
    # get token positions at target_timestep
    # agent who called this must not be in the token
    token_pos_list = [(x, y)
                      for val in token.values()
                      for x, y, t in val['path']
                      if t == target_timestep]
    # count and return free cells
    return sum([1
                for pos in target_neighbours
                if (0 <= pos[0] < input_map.shape[0] and 0 <= pos[1] < input_map.shape[1])
                and input_map[pos] == 0
                and pos not in token_pos_list])


def drop_idle(agent_pool: set[Any],
              curr_agent: Any,
              token: dict[int, dict[str, tuple[int, int] or deque[tuple[int, int, int]]]]
              ) -> dict[int, dict[str, tuple[int, int] or deque[tuple[int, int, int]]]]:
    """
    Drop idle agents' path from the token, excluding calling agent
    :param agent_pool: set of Agent instances
    :param curr_agent: instance of Agent who calls the function
    :param token: summary of other agents planned paths
                  dict -> {agent_name : {'pos': (x,y), ''path': path}}
                  with pos = current agent pos
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...]), future steps
                  x, y -> cartesian coords, t -> timestep
    :return: dict with idle agents path
    """
    idle_token = {}
    for agent in agent_pool:
        if agent.is_idle and agent != curr_agent:
            idle_token[agent.name] = token.pop(agent.name)

    return idle_token
