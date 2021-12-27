import pprint
import random
import unittest
from collections import deque

import numpy as np

from create_dataset.map_creator import create_random_grid_map
from experts.a_star import a_star
from experts.funcs import compute_manhattan_heuristic, is_valid_expansion


# noinspection DuplicatedCode
class AStarTest(unittest.TestCase):

    def test_manhattan_h(self):
        shape = (20, 20)
        size = shape[0]*shape[1]
        density = 0.2
        repetition = 1000

        for _ in range(repetition):
            grid_map = create_random_grid_map(map_shape=shape, map_density=density)
            goal = tuple(np.unravel_index(indices=random.choice(range(size)), shape=shape))
            rand_point = np.unravel_index(indices=random.choice(range(size)), shape=shape)
            h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)

            self.assertIsInstance(h_map, np.ndarray)
            # (abs(rand_point.x – goal.x) + abs(rand_point.y – goal.y)) = h_map[rand_point]
            self.assertEqual((np.abs(rand_point[0] - goal[0]) + np.abs(rand_point[1] - goal[1])),
                             h_map[rand_point])

    # noinspection DuplicatedCode
    def test_is_valid_expansion(self):
        repetition = 10000
        # map creation
        shape = (20, 20)
        density = 0.2
        grid_map = create_random_grid_map(map_shape=shape, map_density=density)

        # get free cell positions
        where_res = np.nonzero(grid_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))

        # get obstacles cell positions
        where_res = np.nonzero(grid_map)
        obs_cell_list = list(zip(where_res[0], where_res[1]))

        # token
        token = {1: [(x, y, t)
                     for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))],
                 2: [(x, y, t)
                     for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))],
                 3: [(x, y, t)
                     for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))]}
        x, y = random.choice(seq=free_cell_list)
        token['stands_still'] = [(x, y, 0)]      # one agent stands still
        # check token validity
        for path in token.values():
            for x, y, t in path:
                self.assertEqual(grid_map[(x, y)], 0)

        '''verify check 1, the cell is inside map boundaries'''
        # child_pos negative
        self.assertFalse(is_valid_expansion(child_pos=(-1, -1), input_map=grid_map, closed_list=None,
                                            parent_pos=None, token=None, child_timestep=None))
        self.assertFalse(is_valid_expansion(child_pos=(0, -1), input_map=grid_map, closed_list=None,
                                            parent_pos=None, token=None, child_timestep=None))
        self.assertFalse(is_valid_expansion(child_pos=(-1, 0), input_map=grid_map, closed_list=None,
                                            parent_pos=None, token=None, child_timestep=None))

        # child_pos out of bounds
        x_bound, y_bound = grid_map.shape
        self.assertFalse(is_valid_expansion(child_pos=(x_bound, y_bound), input_map=grid_map, closed_list=None,
                                            parent_pos=None, token=None, child_timestep=None))
        self.assertFalse(is_valid_expansion(child_pos=(0, y_bound), input_map=grid_map, closed_list=None,
                                            parent_pos=None, token=None, child_timestep=None))
        self.assertFalse(is_valid_expansion(child_pos=(x_bound, 0), input_map=grid_map, closed_list=None,
                                            parent_pos=None, token=None, child_timestep=None))

        '''verify check 2, the cell has already been expanded'''
        # next pos, inbound
        child_pos = random.choice(free_cell_list)
        # set closed list, child_pos already expanded
        closed_list = np.zeros(grid_map.shape, dtype=int)
        closed_list[child_pos] = 1

        self.assertFalse(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                            parent_pos=None, token=None, child_timestep=None))

        '''verify check 3, the cell has an obstacle inside'''
        # next pos, inbound but with obstacle
        child_pos = random.choice(obs_cell_list)
        # set closed list, child_pos not expanded
        closed_list = np.zeros(grid_map.shape, dtype=int)

        self.assertFalse(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                            parent_pos=None, token=None, child_timestep=None))

        '''verify classic A*'''
        # next pos, inbound and valid
        child_pos = random.choice(free_cell_list)
        # set closed list, child_pos not expanded
        closed_list = np.zeros(grid_map.shape, dtype=int)

        self.assertTrue(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                           parent_pos=None, token=None, child_timestep=None))
        # check it defaults to classic A* if one of parent_pos, token or child_timestep is missing
        self.assertTrue(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                           parent_pos=(0, 0), token=token, child_timestep=None))
        self.assertTrue(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                           parent_pos=(0, 0), token=None, child_timestep=3))
        self.assertTrue(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                           parent_pos=None, token=token, child_timestep=3))

        '''verify check 4'''
        closed_list = np.zeros(grid_map.shape, dtype=int)

        # a) check no swap constraint
        # pick a parent node such that exists a path in the token that will go there at child_timestep
        for _ in range(repetition):
            child_timestep = random.choice(range(1, 5))     # child_ts needs to be guaranteed > 0
            swap = random.choice([(step[:-1], (path[child_timestep-1][:-1]))
                                  for path in token.values()
                                  for step in path
                                  if step[-1] == child_timestep])
            child_pos, parent_pos = swap

            self.assertFalse(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                                parent_pos=parent_pos, token=token, child_timestep=child_timestep))

        # b) check no node collision
        for _ in range(repetition):
            child_timestep = random.choice(range(1, 5))  # child_ts needs to be guaranteed > 0
            child_pos = random.choice([(x, y)
                                       for path in token.values()
                                       for x, y, t in path
                                       if t == child_timestep])
            parent_pos = random.choice(free_cell_list)  # doesn't matter for this check

            self.assertFalse(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                                parent_pos=parent_pos, token=token, child_timestep=child_timestep))

        # c) check avoid planning into agents standing still
        child_timestep = random.choice(range(1, 5))  # child_ts needs to be guaranteed > 0
        child_pos = token['stands_still'][0][:-1]
        parent_pos = random.choice(free_cell_list)  # doesn't matter for this check

        self.assertFalse(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                            parent_pos=parent_pos, token=token, child_timestep=child_timestep))

        '''true, complete execution'''
        closed_list = np.zeros(grid_map.shape, dtype=int)
        for _ in range(repetition):
            child_timestep = random.choice(range(1, 5))  # child_ts needs to be guaranteed > 0
            # no token positions
            token_pos_list = [(x, y)
                              for path in token.values()
                              for x, y, t in path]
            child_pos = random.choice(list(set(free_cell_list) - set(token_pos_list)))
            parent_pos = random.choice(list(set(free_cell_list)-set(token_pos_list)))

            self.assertTrue(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                               parent_pos=parent_pos, token=token, child_timestep=child_timestep))

            # yes token, but no collision nor swapping
            token_pos_list = [(x, y)
                              for path in token.values()
                              for x, y, t in path
                              if t == child_timestep or t == 0]
            child_pos = random.choice(list(set(free_cell_list) - set(token_pos_list)))
            parent_pos = random.choice(list(set(free_cell_list) - set(token_pos_list)))

            self.assertTrue(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list,
                                               parent_pos=parent_pos, token=token, child_timestep=child_timestep))

    def test_a_star(self):
        repetition = 1000
        # map creation
        shape = (20, 20)
        density = 0.2
        grid_map = create_random_grid_map(map_shape=shape, map_density=density)

        # get free cell positions
        where_res = np.nonzero(grid_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))

        # token
        token = {1: [(x, y, t)
                     for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))],
                 2: [(x, y, t)
                     for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))],
                 3: [(x, y, t)
                     for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))]}
        x, y = random.choice(seq=free_cell_list)
        token['stands_still'] = [(x, y, 0)]  # one agent stands still

        token_step_list = [step
                           for path in token.values()
                           for step in path]
        token_pos_list = [(x, y)
                          for path in token.values()
                          for x, y, t in path]
        # check token validity
        for x, y, t in token_step_list:
            self.assertEqual(grid_map[(x, y)], 0)

        '''Classic A*, no token nor given, precomputed heuristic'''
        for _ in range(repetition):
            # start_pos, goal and heuristic
            start_pos = random.choice(free_cell_list)
            goal = random.choice(list(set(free_cell_list) - set(start_pos)))

            try:
                path, length = a_star(input_map=grid_map,
                                      start=start_pos, goal=goal,
                                      token=None, h_map=None)

                self.assertIsInstance(length, int)
                self.assertIsInstance(path, deque)
                for step in path:
                    self.assertIsInstance(step, tuple)
                self.assertEqual(length, len(path))

            except ValueError:
                pass

        '''A* with given, precomputed heuristic'''
        for _ in range(repetition):
            # start_pos, goal and heuristic
            start_pos = random.choice(free_cell_list)
            goal = random.choice(list(set(free_cell_list) - set(start_pos)))
            h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)

            try:
                path, length = a_star(input_map=grid_map,
                                      start=start_pos, goal=goal,
                                      token=None, h_map=h_map)

                self.assertIsInstance(length, int)
                self.assertIsInstance(path, deque)
                for step in path:
                    self.assertIsInstance(step, tuple)
                self.assertEqual(length, len(path))

            except ValueError:
                pass

        '''A* with token'''
        for _ in range(repetition):
            # start_pos, goal and heuristic
            start_pos = random.choice(list(set(free_cell_list) - set(token_pos_list)))
            goal = random.choice(list(set(free_cell_list) - set(start_pos)))

            try:
                path, length = a_star(input_map=grid_map,
                                      start=start_pos, goal=goal,
                                      token=token, h_map=None)

                self.assertIsInstance(length, int)
                self.assertIsInstance(path, deque)
                for step in path:
                    self.assertIsInstance(step, tuple)
                    self.assertNotIn(step, token_step_list)
                self.assertEqual(length, len(path))

            except ValueError:
                pass

        '''A* with token and given h_map'''
        printed = False
        for _ in range(repetition):
            # start_pos, goal and heuristic
            start_pos = random.choice(list(set(free_cell_list) - set(token_pos_list)))
            goal = random.choice(list(set(free_cell_list) - set(start_pos)))
            h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)

            try:
                path, length = a_star(input_map=grid_map,
                                      start=start_pos, goal=goal,
                                      token=token, h_map=h_map)

                self.assertIsInstance(length, int)
                self.assertIsInstance(path, deque)
                for step in path:
                    self.assertIsInstance(step, tuple)
                    self.assertNotIn(step, token_step_list)
                self.assertEqual(length, len(path))

                if not printed:
                    print('Map:')
                    pprint.pprint(grid_map)
                    print('\nToken:')
                    pprint.pprint(token)
                    print(f'\nStart: {start_pos}, Goal: {goal}')
                    print(f'\nA* path:')
                    pprint.pprint(path)
                    printed = True

            except ValueError:
                pass


if __name__ == '__main__':
    unittest.main()
