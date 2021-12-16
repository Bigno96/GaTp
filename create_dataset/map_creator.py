"""
Dataset creation utility file
Functions and classes to generate maps filled with obstacles
Map types: random_grid
"""

import numpy as np


class MapCreator:
    """
    Class for creating maps and filling it with obstacles, according to density and complexity
    Different types of map supported: random_grid
    Outside entry point -> create_{map_type} functions
    """

    def __init__(self, map_size=(20, 20), map_density=0.2):
        """
        Instantiate the class
        :param map_size: [H, W], int, size of the map
        :param map_density: float, range [0:1], percentage of obstacles in the map
        """
        self.__h = map_size[0]
        self.__w = map_size[1]
        self.__density = map_density

    def create_random_grid_map(self):
        """
        Create random grid type map with randomly placed obstacles
        :return: map: np.ndarray -> '1' in obstacles places, '0' elsewhere
        """
        num_cells = int(self.__h * self.__w)
        num_obstacles = int(num_cells * self.__density)
        flat_map = np.zeros(num_cells, dtype=np.int8)       # array of zero, dim: h*w
        # get a random permutation of numbers between 0 (included) and 'num_cells' (excluded)
        p = np.random.permutation(num_cells)
        # set to 1 the elements give by the first 'num_obstacles' indexes of the permutation
        flat_map[p[:num_obstacles]] = 1
        # reshape as matrix
        map_ = flat_map.reshape(self.__h, self.__w)

        return map_
