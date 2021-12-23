import unittest

import numpy as np

from create_dataset.map_creator import create_random_grid_map


class MapCreatorTest(unittest.TestCase):

    def test_random_grid_map(self):
        shape = (20, 20)
        size = shape[0] * shape[1]
        grid_map = None
        repetition = 10000

        '''normal density'''
        density = 0.2
        for _ in range(repetition):
            grid_map = create_random_grid_map(map_shape=shape, map_density=density)

            self.assertEqual(size*density, np.count_nonzero(grid_map))
            self.assertIsInstance(grid_map, np.ndarray)
        print(f'Random_grid_map, shape={shape}, density={density}\n'
              f'{grid_map}')

        '''no obstacles'''
        density = 0
        grid_map = create_random_grid_map(map_shape=shape, map_density=density)

        self.assertEqual(0, np.count_nonzero(grid_map))

        '''only obstacles'''
        density = 1
        grid_map = create_random_grid_map(map_shape=shape, map_density=density)

        self.assertEqual(size, np.count_nonzero(grid_map))


if __name__ == '__main__':
    unittest.main()
