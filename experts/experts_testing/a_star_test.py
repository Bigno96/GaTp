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
        self.assertFalse(is_valid_expansion(child_pos=(-1, -1), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(child_pos=(0, -1), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(child_pos=(-1, 0), input_map=grid_map, closed_list=None))

        # child_pos out of positive bounds
        x_bound, y_bound = grid_map.shape
        self.assertFalse(is_valid_expansion(child_pos=(x_bound, y_bound), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(child_pos=(0, y_bound), input_map=grid_map, closed_list=None))
        self.assertFalse(is_valid_expansion(child_pos=(x_bound, 0), input_map=grid_map, closed_list=None))

        '''verify check 2, the cell has already been expanded'''
        # next pos, inbound
        child_pos = random.choice(free_cell_list)
        # set closed list, child_pos already expanded
        closed_list = np.zeros(grid_map.shape, dtype=int)
        closed_list[child_pos] = 1

        self.assertFalse(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list))

        '''verify check 3, the cell has an obstacle inside'''
        # next pos, inbound but with obstacle
        child_pos = random.choice(obs_cell_list)
        # set closed list, child_pos not expanded
        closed_list = np.zeros(grid_map.shape, dtype=int)

        self.assertFalse(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list))

        '''verify return True'''
        # next pos, inbound and valid
        child_pos = random.choice(free_cell_list)
        # set closed list, child_pos not expanded
        closed_list = np.zeros(grid_map.shape, dtype=int)

        self.assertTrue(is_valid_expansion(child_pos=child_pos, input_map=grid_map, closed_list=closed_list))

    def test_check_token_conflicts(self):

        '''check it defaults to true if some parameter is missing'''
        # default behaviour for classic A* usage
        self.assertTrue(check_token_conflicts(token=None, new_pos=(2, 2), curr_pos=(0, 0), new_timestep=0))
        self.assertTrue(check_token_conflicts(token={1: [(2, 2, 0)]}, new_pos=None, curr_pos=(0, 0), new_timestep=0))
        self.assertTrue(check_token_conflicts(token={1: [(2, 2, 0)]}, new_pos=(2, 2), curr_pos=None, new_timestep=0))
        self.assertTrue(check_token_conflicts(token={1: [(2, 2, 0)]}, new_pos=(2, 2), curr_pos=(0, 0), new_timestep=None))

        '''check no swap constraint'''
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_timestep = 1
        new_pos = (4, 5)
        curr_pos = (4, 6)

        self.assertFalse(check_token_conflicts(token=token, new_pos=new_pos, curr_pos=curr_pos,
                                               new_timestep=new_timestep))

        '''check no nodes conflicts'''
        # crash into a moving agent, avoided
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_timestep = 1
        new_pos = (4, 6)
        curr_pos = (3, 6)

        self.assertFalse(check_token_conflicts(token=token, new_pos=new_pos, curr_pos=curr_pos,
                                               new_timestep=new_timestep))

        # goes into a spot before the other agent, permitted
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2), (5, 7, 3)]}
        new_timestep = 1
        new_pos = (5, 6)
        curr_pos = (4, 6)

        self.assertTrue(check_token_conflicts(token=token, new_pos=new_pos, curr_pos=curr_pos,
                                               new_timestep=new_timestep))

        '''test boundaries'''
        # see behaviour when new_pos == curr_pos
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_timestep = 1
        new_pos = (4, 5)
        curr_pos = (4, 5)

        self.assertTrue(check_token_conflicts(token=token, new_pos=new_pos, curr_pos=curr_pos,
                                              new_timestep=new_timestep))

        # called at the start of the system
        token = {0: [(4, 5, 0), (4, 6, 1), (5, 6, 2)]}
        new_timestep = 0
        new_pos = (4, 6)
        curr_pos = (4, 6)

        self.assertTrue(check_token_conflicts(token=token, new_pos=new_pos, curr_pos=curr_pos,
                                               new_timestep=new_timestep))

        new_timestep = 0
        new_pos = (4, 5)
        curr_pos = (4, 5)

        self.assertFalse(check_token_conflicts(token=token, new_pos=new_pos, curr_pos=curr_pos,
                                               new_timestep=new_timestep))

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
            # start_pos, goal
            start_pos, goal = random.sample(population=free_cell_list, k=2)
            # heuristic
            h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)

            try:
                # no heuristic
                path1, length1 = a_star(input_map=grid_map,
                                        start=start_pos, goal=goal)

                # precomputed heuristic
                path2, length2 = a_star(input_map=grid_map,
                                        start=start_pos, goal=goal,
                                        h_map=h_map)

                # no heuristic, starting timestep > 0
                starting_timestep = random.choice(range(1, 100))
                path3, length3 = a_star(input_map=grid_map,
                                        start=start_pos, goal=goal,
                                        starting_t=starting_timestep)

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
                # start from start_pos, timestep 0
                self.assertEqual(path1[0][:-1], start_pos)
                self.assertEqual(path1[0][-1], 0)
                # ends in goal at timestep = length-1
                self.assertEqual(path1[-1][:-1], goal)
                self.assertEqual(path1[-1][-1], length1-1)

                # checking consistency with different starting time
                self.assertEqual(path3[0][:-1], start_pos)
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
            # start_pos, goal
            start_pos, goal = random.sample(population=(list(set(free_cell_list) - set(token_pos_list))),
                                            k=2)
            # heuristic
            h_map = compute_manhattan_heuristic(input_map=grid_map, goal=goal)
            # starting time
            starting_timestep = random.choice(range(tok_path_length))

            try:
                # no heuristic
                path1, length1 = a_star(input_map=grid_map,
                                        start=start_pos, goal=goal,
                                        token=token, starting_t=starting_timestep)

                start_time = timeit.default_timer()
                # as in TP, with token and heuristic
                path2, length2 = a_star(input_map=grid_map,
                                        start=start_pos, goal=goal,
                                        token=token, h_map=h_map, starting_t=starting_timestep)
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
                # start from start_pos, timestep == starting_time
                # ends in goal at timestep = length-1
                self.assertEqual(path1[0][:-1], start_pos)
                self.assertEqual(path1[-1][:-1], goal)
                self.assertEqual(path1[0][-1], starting_timestep)
                self.assertEqual(path1[-1][-1], length1+starting_timestep-1)

                if not printed:
                    print('Map:')
                    pprint(grid_map)
                    print('\nToken:')
                    pprint(token)
                    print(f'\nStart: {start_pos}, Goal: {goal}')
                    print(f'\nA* path:')
                    pprint(path1)
                    printed = True

            except ValueError as err:
                print(err)
                pass

        print(f'\nAverage full A* execution time: {statistics.mean(time_list)}')


def get_grid_map_free_cell_token(shape, density, agent_num, token_path_length):
    # map creation
    grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

    # get free cell positions
    where_res = np.nonzero(grid_map == 0)
    free_cell_list = list(zip(where_res[0], where_res[1]))

    # cell pool, avoid repetition
    pool = random.sample(population=free_cell_list, k=int(token_path_length*3)+1)

    # token
    token = {}
    for i in range(agent_num-1):
        token[i] = [(x, y, t)
                    for t, (x, y) in enumerate(pool[int(i*token_path_length):
                                                    int((i+1)*token_path_length)])
                    ]
    x, y = pool[-1]
    token['stands_still'] = [(x, y, 0)]  # one agent stands still

    return grid_map, free_cell_list, token


if __name__ == '__main__':
    unittest.main()