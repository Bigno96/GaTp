import random
import statistics
import timeit
import unittest
from collections import deque
from pprint import pprint

import numpy as np
from scipy.spatial.distance import cityblock

from create_dataset.map_creator import create_random_grid_map
from experts.a_star import a_star
from utils.expert_utils import compute_manhattan_heuristic, is_valid_expansion, check_token_conflicts
from testing.test_utils import get_grid_map_free_cell_token


# noinspection PySingleQuotedDocstring
class AStarTest(unittest.TestCase):

    def test_manhattan_h(self):
        shape = (20, 20)
        size = shape[0]*shape[1]
        density = 0.2

        grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)
        goal = np.unravel_index(indices=random.choice(range(size)), shape=shape)
        rand_point = np.unravel_index(indices=random.choice(range(size)), shape=shape)
        h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)

        self.assertIsInstance(h_map, np.ndarray)
        # (abs(rand_point.x – goal.x) + abs(rand_point.y – goal.y)) = h_map[rand_point]
        self.assertEqual(cityblock(rand_point, goal), h_map[rand_point])

    def test_is_valid_expansion(self):
        shape = (20, 20)
        density = 0.2
        grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

        # get free cell positions
        where_res = np.nonzero(grid_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))

        # get obstacles cell positions
        where_res = np.nonzero(grid_map)
        obs_cell_list = list(zip(where_res[0], where_res[1]))

        '''verify check 1, the cell is inside map boundaries'''
        # child_pos negative
        self.assertFalse(is_valid_expansion(next_node=(-1, -1, 0), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(next_node=(0, -1, 0), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(next_node=(-1, 0, 0), input_map=grid_map, closed_list=None))

        # child_pos out of positive bounds
        x_bound, y_bound = grid_map.shape
        self.assertFalse(is_valid_expansion(next_node=(x_bound, y_bound, 0), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(next_node=(0, y_bound, 0), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(next_node=(x_bound, 0, 0), input_map=grid_map, closed_list=None))

        '''verify check 2, the cell has already been expanded'''
        # next pos, inbound
        child_pos = random.choice(free_cell_list)
        # set closed list, child_pos already expanded
        child_node = (child_pos[0], child_pos[1], 0)
        closed_list = {child_node}

        self.assertFalse(is_valid_expansion(next_node=child_node, input_map=grid_map, closed_list=closed_list))

        '''verify check 3, the cell has an obstacle inside'''
        # next pos, inbound but with obstacle
        child_pos = random.choice(obs_cell_list)
        # set closed list, child_pos not expanded
        child_node = (child_pos[0], child_pos[1], 0)
        closed_list = set()

        self.assertFalse(is_valid_expansion(next_node=child_node, input_map=grid_map, closed_list=closed_list))

        '''verify return True'''
        # next pos, inbound and valid
        child_pos = random.choice(free_cell_list)
        # set closed list, child_pos not expanded
        child_node = (child_pos[0], child_pos[1], 0)
        closed_list = set()

        self.assertTrue(is_valid_expansion(next_node=child_node, input_map=grid_map, closed_list=closed_list))

    def test_check_token_conflicts(self):

        '''check it defaults to true if some parameter is missing'''
        # default behaviour for classic A* usage
        self.assertTrue(check_token_conflicts(token=None, next_node=(2, 2, 0), curr_node=(0, 0, 0)))
        self.assertTrue(check_token_conflicts(token={1: [(2, 2, 0)]}, next_node=None, curr_node=(0, 0, 0)))
        self.assertTrue(check_token_conflicts(token={1: [(2, 2, 0)]}, next_node=(2, 2, 0), curr_node=None))

        '''check no swap constraint'''
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_node = (4, 5, 1)
        curr_node = (4, 6, 0)

        self.assertFalse(check_token_conflicts(token=token, next_node=new_node, curr_node=curr_node))

        '''check no nodes conflicts'''
        # crash into a moving agent, avoided
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_node = (4, 6, 1)
        curr_node = (3, 6, 0)

        self.assertFalse(check_token_conflicts(token=token, next_node=new_node, curr_node=curr_node))

        # goes into a spot before the other agent, permitted
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2), (5, 7, 3)]}
        new_node = (5, 6, 1)
        curr_node = (4, 6, 0)

        self.assertTrue(check_token_conflicts(token=token, next_node=new_node, curr_node=curr_node))

        '''test boundaries'''
        # see behaviour when new_pos == curr_pos
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_node = (4, 5, 1)
        curr_node = (4, 5, 1)

        self.assertTrue(check_token_conflicts(token=token, next_node=new_node, curr_node=curr_node))

        # called at the start of the system
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_node = (4, 6, 0)
        curr_node = (4, 6, -1)

        self.assertTrue(check_token_conflicts(token=token, next_node=new_node, curr_node=curr_node))

        new_node = (4, 5, 0)
        curr_node = (4, 5, -1)

        self.assertFalse(check_token_conflicts(token=token, next_node=new_node, curr_node=curr_node))

    def test_a_star(self):
        repetition = 1000
        shape = (20, 20)
        density = 0.2

        agent_num = 5
        '''Classic A*, no token'''
        for _ in range(repetition):
            grid_map, free_cell_list, t_ = get_grid_map_free_cell_token(shape=shape, density=density,
                                                                        agent_num=agent_num,
                                                                        token_path_length=0)
            # start, goal
            start, goal = random.sample(population=free_cell_list, k=2)
            # heuristic
            h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)

            try:
                # no heuristic
                path1, length1 = a_star(input_map=grid_map,
                                        start=start, goal=goal, include_start_node=True)

                # precomputed heuristic
                path2, length2 = a_star(input_map=grid_map,
                                        start=start, goal=goal,
                                        h_map=h_map, include_start_node=True)

                # no heuristic, starting timestep > 0
                starting_timestep = random.choice(range(1, 100))
                path3, length3 = a_star(input_map=grid_map,
                                        start=start, goal=goal,
                                        starting_t=starting_timestep, include_start_node=True)

                # check type integrity
                self.assertIsInstance(length1, int)
                self.assertIsInstance(length2, int)
                self.assertIsInstance(length3, int)
                self.assertIsInstance(path1, deque)
                self.assertIsInstance(path2, deque)
                self.assertIsInstance(path3, deque)
                for step in path1:
                    self.assertIsInstance(step, tuple)
                for step in path2:
                    self.assertIsInstance(step, tuple)
                for step in path3:
                    self.assertIsInstance(step, tuple)

                # check return value integrity
                self.assertEqual(length1, len(path1))
                self.assertEqual(length2, len(path2))
                self.assertEqual(length3, len(path3))

                # equal with or without precomputed h
                self.assertEqual(length1, length2)
                self.assertEqual(path1, path2)
                # start from start, timestep 0
                self.assertEqual(path1[0][:-1], start)
                self.assertEqual(path1[0][-1], 0)
                # ends in goal at timestep = length-1
                self.assertEqual(path1[-1][:-1], goal)
                self.assertEqual(path1[-1][-1], length1-1)

                # checking consistency with different starting time
                self.assertEqual(path3[0][:-1], start)
                self.assertEqual(path3[0][-1], starting_timestep)
                self.assertEqual(path3[-1][:-1], goal)
                self.assertEqual(path3[-1][-1], length3+starting_timestep-1)

                # without timestep, equal to the other two
                no_timestep_path1 = [(x, y) for x, y, t_ in path1]
                no_timestep_path3 = [(x, y) for x, y, t_ in path3]
                self.assertEqual(no_timestep_path1, no_timestep_path3)

            except ValueError as err:
                print(err)
                pass

        '''A* with token'''
        printed = False
        time_list = []
        tok_path_length = 10
        agent_num = 5
        for _ in range(repetition):
            grid_map, free_cell_list, token = get_grid_map_free_cell_token(shape=shape, density=density,
                                                                           agent_num=agent_num,
                                                                           token_path_length=tok_path_length)
            token_step_list = [step
                               for path in token.values()
                               for step in path]
            token_pos_list = [(x, y)
                              for path in token.values()
                              for x, y, t in path]
            # start, goal
            start, goal = random.sample(population=(list(set(free_cell_list) - set(token_pos_list))),
                                            k=2)
            # heuristic
            h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)
            # starting time
            starting_timestep = random.choice(range(tok_path_length))

            try:
                # no heuristic
                path1, length1 = a_star(input_map=grid_map,
                                        start=start, goal=goal,
                                        token=token, starting_t=starting_timestep,
                                        include_start_node=False)

                start_time = timeit.default_timer()
                # as in TP, with token and heuristic
                path2, length2 = a_star(input_map=grid_map,
                                        start=start, goal=goal,
                                        token=token, h_map=h_map, starting_t=starting_timestep,
                                        include_start_node=False)
                diff_time = timeit.default_timer() - start_time
                time_list.append(diff_time)

                # check type integrity
                self.assertIsInstance(length1, int)
                self.assertIsInstance(length2, int)
                self.assertIsInstance(path1, deque)
                self.assertIsInstance(path2, deque)
                for step in path1:
                    self.assertIsInstance(step, tuple)
                    self.assertNotIn(step, token_step_list)     # check not in token
                self.assertEqual(length1, len(path1))
                self.assertEqual(length2, len(path2))

                # equal with or without precomputed h
                self.assertEqual(path1, path2)
                # start from timestep == starting_time
                # ends in goal at timestep = length-1
                self.assertEqual(path1[-1][:-1], goal)
                self.assertEqual(path1[0][-1], starting_timestep)
                self.assertEqual(path1[-1][-1], length1+starting_timestep-1)

                if not printed:
                    print('Map:')
                    pprint(grid_map)
                    print('\nToken:')
                    pprint(token)
                    print(f'\nStart: {start}, Goal: {goal}')
                    print(f'Starting timestep: {starting_timestep}')
                    print(f'\nA* path:')
                    pprint(path1)
                    printed = True

            except ValueError as err:
                print(err)
                pass

        print(f'\nAverage full A* execution time: {statistics.mean(time_list)}')


if __name__ == '__main__':
    unittest.main()
