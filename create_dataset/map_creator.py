"""
Dataset creation utility file
Functions to generate maps filled with obstacles
Map types supported:
            - random_grid
"""

import numpy as np


def create_random_grid_map(map_shape, map_density):
    """
    Create random grid type map with randomly placed obstacles
    :param map_shape: [H, W], int, size of the map
    :param map_density: float, range [0, 1], percentage of obstacles in the map
    :return: map: np.ndarray, shape=(H, W) -> '1' in obstacles places, '0' elsewhere
    """
    num_cells = int(map_shape[0] * map_shape[1])
    num_obstacles = int(num_cells * map_density)
    flat_map = np.zeros(num_cells, dtype=np.int8)  # array of zero, dim: h*w
    # get a random permutation of numbers between 0 (included) and 'num_cells' (excluded)
    p = np.random.permutation(num_cells)
    # set to 1 the elements give by the first 'num_obstacles' indexes of the permutation
    flat_map[p[:num_obstacles]] = 1
    # reshape as matrix
    return flat_map.reshape(map_shape)
