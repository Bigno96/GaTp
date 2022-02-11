import statistics
import timeit
import unittest
from pprint import pprint

import numpy as np

from create_dataset.map_creator import create_random_grid_map, is_connected
from experts.a_star import a_star
from testing.test_utils import create_grid_test_map, find_start_goal
from utils.expert_utils import NoPathError


class MapCreatorTest(unittest.TestCase):

    def test_is_connected(self):
        shape = (20, 20)
        size = shape[0] * shape[1]
        repetition = 1000
        time_list = []

        density = 0.2
        obstacle_count = int(size * density)
        for _ in range(repetition):
            grid_map = create_grid_test_map(shape=shape, density=density)

            # pick start pos and goal
            start, goal = find_start_goal(input_map=grid_map)

            start_time = timeit.default_timer()

            is_conn = is_connected(input_map=grid_map, size=size, obstacle_count=obstacle_count)

            time_diff = timeit.default_timer() - start_time
            time_list.append(time_diff)

            try:
                a_star(input_map=grid_map, start=start, goal=goal, include_start_node=True)
            except NoPathError:
                if is_conn:     # path not found in a connected map
                    pprint(grid_map)
                    self.fail("a_star didn't find path in connected map")
                else:
                    pass

        print(f'is_connected execution time: {statistics.mean(time_list)}')

        '''Limit testing'''
        # no obstacles
        density = 0
        obstacle_count = int(size * density)
        grid_map = create_grid_test_map(shape=shape, density=density)

        self.assertTrue(is_connected(input_map=grid_map, size=size, obstacle_count=obstacle_count))

        # all obstacles
        density = 1
        obstacle_count = int(size * density)
        grid_map = create_grid_test_map(shape=shape, density=density)

        self.assertFalse(is_connected(input_map=grid_map, size=size, obstacle_count=obstacle_count))

    def test_random_grid_map(self):
        shape = (20, 20)
        size = shape[0] * shape[1]
        grid_map = None
        repetition = 1000

        '''normal density'''
        density = 0.2
        for _ in range(repetition):
            grid_map = create_random_grid_map(map_shape=shape, map_density=density,
                                              connected=True)

            self.assertEqual(size*density, np.count_nonzero(grid_map))
            self.assertIsInstance(grid_map, np.ndarray)

            # pick start pos and goal
            start, goal = find_start_goal(input_map=grid_map)

            try:
                a_star(input_map=grid_map, start=start, goal=goal, include_start_node=True)
            except NoPathError:
                pprint(grid_map)
                self.fail("a_star didn't find path in connected map")

        print(f'Random_grid_map, shape={shape}, density={density}\n'
              f'{grid_map}')

        '''no obstacles, doesn't matter connection'''
        density = 0
        grid_map = create_random_grid_map(map_shape=shape, map_density=density,
                                          connected=True)

        self.assertEqual(0, np.count_nonzero(grid_map))

        grid_map = create_random_grid_map(map_shape=shape, map_density=density,
                                          connected=False)

        self.assertEqual(0, np.count_nonzero(grid_map))

        '''only obstacles, doesn't matter connection'''
        density = 1
        grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

        self.assertEqual(size, np.count_nonzero(grid_map))

        grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=False)

        self.assertEqual(size, np.count_nonzero(grid_map))


if __name__ == '__main__':
    unittest.main()
