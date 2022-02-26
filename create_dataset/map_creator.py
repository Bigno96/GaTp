"""
Dataset creation utility file
Functions to generate maps filled with obstacles
Map types supported:
            - random_grid
"""

import random

import numpy as np

from collections import deque
from typing import Optional

# moves dictionary
DELTA = [(-1, 0),  # go up
         (0, -1),  # go left
         (1, 0),  # go down
         (0, 1)]  # go right


def create_random_grid_map(map_shape: tuple[int, int],
                           map_density: float,
                           connected: bool
                           ) -> np.array:
    """
    Create connected, random grid type map with randomly placed obstacles
    :param map_shape: (H, W), size of the map
    :param map_density: range [0, 1], percentage of obstacles in the map
    :param connected: whether to generate connected map or not
    :return: map: shape=(H, W) -> '1' in obstacles places, '0' elsewhere
    """
    cell_count = int(map_shape[0] * map_shape[1])
    obstacle_count = int(cell_count * map_density)
    flat_map = np.zeros(cell_count, dtype=np.int8)  # array of zero, dim: h*w

    while True:
        # get a random permutation of numbers between 0 (included) and 'cell_count' (excluded)
        p = np.random.permutation(cell_count)
        # set to 1 the elements give by the first 'obstacle_count' indexes of the permutation
        flat_map[p[:obstacle_count]] = 1
        # reshape as matrix
        grid_map = flat_map.reshape(map_shape)

        # if it doesn't need to be connected or has none/only obstacles
        if not connected or cell_count == obstacle_count or obstacle_count == 0:
            return grid_map

        # check if it's connected
        if not is_connected(input_map=grid_map,
                            size=cell_count,
                            obstacle_count=obstacle_count):
            # reset obstacles and try again
            flat_map[p[:obstacle_count]] = 0
        else:
            return grid_map


def is_connected(input_map: np.array,
                 size: int,
                 obstacle_count: int
                 ) -> bool:
    """
    Check if all the free cells are connected
    :param input_map: shape=(H, W) -> '1' in obstacles places, '0' elsewhere
    :param size: total number of cells in the map
    :param obstacle_count: number of obstacles in the map
    :return: True if all free cells are connected, False else
    """
    free_cell_count = size - obstacle_count

    # no free cell -> not connected
    if not free_cell_count:
        return False

    # all free cells -> is connected
    if free_cell_count == size:
        return True

    visited = set()
    # pick starting point
    where_res = np.nonzero(input_map == 0)
    free_cell_list = list(zip(where_res[0], where_res[1]))
    starting_node = random.choice(free_cell_list)
    # create adjacency list of the map
    graph = create_adj_list(input_map=input_map,
                            free_cell_list=free_cell_list)
    # depth first search adding nodes to visited set
    dfs(visited=visited,
        graph=graph,
        start_node=starting_node)

    # count how many nodes were visited
    visit_count = len(visited)
    return visit_count == free_cell_count


def create_adj_list(input_map: np.array,
                    free_cell_list: list[tuple[int, int]]
                    ) -> dict[tuple[int, int], Optional[list[tuple[int, int]]]]:
    """
    Create adjacency list for graph representation of input_map
    :param input_map: shape=(H, W) -> '1' in obstacles places, '0' elsewhere
    :param free_cell_list: free cell coordinates
           [(x,y), ...]
    :return: dict, graph represented with adjacency list implemented through python dict
             {(x,y) : [(x1, y1), (x2, y2), ...]}
    """
    graph = {}
    x_m, y_m = input_map.shape

    # loop all free cells
    for cell in free_cell_list:
        # get neighbours inbound
        neighbours = [(cell[0] + move[0], cell[1] + move[1])
                      for move in DELTA]
        neighbours = [(x, y)
                      for x, y in neighbours
                      if x_m > x >= 0 and y_m > y >= 0 == input_map[(x, y)]]
        # add dict entry, copy to avoid bad referencing
        graph[cell] = neighbours.copy()

    return graph


def dfs(visited: set[tuple[int, int]],
        graph: dict[tuple[int, int], Optional[list[tuple[int, int]]]],
        start_node: tuple[int, int]) -> None:
    """
    Implement depth first search on an adjacency list
    :param visited: already visited nodes
    :param graph: dict, graph represented with adjacency list implemented through python dict
                  {(x,y) : [(x1, y1), (x2, y2), ...]}
    :param start_node: node to start from
    """
    stack = deque()
    stack.append(start_node)

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)   # visit it
            stack.extend(graph[node])   # add all neighbours
