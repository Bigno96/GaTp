import unittest
from random import sample

import numpy as np

from create_dataset.map_creator import create_random_grid_map
from create_dataset.scenario_creator import create_task, create_starting_pos


class ScenarioCreatorTest(unittest.TestCase):

    def test_create_starting_pos(self):
        shape = (20, 20)
        size = shape[0] * shape[1]
        map_density = 0.2
        grid_map = create_random_grid_map(map_shape=shape, map_density=map_density)
        agent_number = 10
        repetition = 10000
        # get free cell positions
        where_res = np.nonzero(grid_map == 0)
        free_cell_list = list(zip(where_res[0], where_res[1]))

        '''random pos list'''
        for _ in range(repetition):
            start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                                 mode='random', fixed_pos_list=None)

            self.assertIsInstance(start_pos_list, list)
            self.assertEqual(agent_number, len(start_pos_list))
            self.assertEqual(agent_number, len(set(start_pos_list)))  # no repeating positions
            for pos in start_pos_list:
                self.assertIsInstance(pos, tuple)
                self.assertEqual(0, grid_map[pos])      # no obstacles

        '''limit case, no agents'''
        agent_number = 0
        start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                             mode='random', fixed_pos_list=None)

        self.assertFalse(start_pos_list)

        '''limit case, full grid'''
        agent_number = int(size * (1-map_density))
        start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                             mode='random', fixed_pos_list=None)

        self.assertEqual(agent_number, len(start_pos_list))
        self.assertEqual(agent_number, len(set(start_pos_list)))  # no repeating positions

        '''more agents than free cells'''
        agent_number = size
        self.assertRaises(ValueError,
                          lambda: create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                                      mode='random', fixed_pos_list=None))

        '''fixed position mode, with exactly enough spots for all the agents'''
        agent_number = 10
        for _ in range(repetition):
            fixed_pos = sample(population=free_cell_list, k=agent_number)    # list of tuples
            start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                                 mode='fixed', fixed_pos_list=fixed_pos)

            self.assertIsInstance(start_pos_list, list)
            for pos in start_pos_list:
                self.assertIsInstance(pos, tuple)
                self.assertEqual(0, grid_map[pos])  # no obstacles
            self.assertEqual(agent_number, len(start_pos_list))
            self.assertEqual(agent_number, len(set(start_pos_list)))  # no repeating positions

        '''fixed position mode, with more than necessary spots for all the agents'''
        for _ in range(repetition):
            fixed_pos = sample(population=free_cell_list, k=agent_number+1)  # list of tuples
            start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                                 mode='fixed', fixed_pos_list=fixed_pos)

            for pos in start_pos_list:
                self.assertEqual(0, grid_map[pos])  # no obstacles
            self.assertEqual(agent_number, len(start_pos_list))
            self.assertEqual(agent_number, len(set(start_pos_list)))  # no repeating positions

        '''fixed position mode, with not enough spots for all the agents'''
        fixed_pos = sample(population=free_cell_list, k=agent_number-1)  # list of tuples
        self.assertRaises(ValueError,
                          lambda: create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                                      mode='fixed', fixed_pos_list=fixed_pos))

    def test_create_task(self):
        shape = (20, 20)
        map_density = 0.2
        grid_map = create_random_grid_map(map_shape=shape, map_density=map_density)
        agent_number = 10
        repetition = 10000
        start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_number,
                                             mode='random', fixed_pos_list=None)
        non_task_ep_list = start_pos_list.copy()

        '''free mode'''
        task_list = []      # while testing free, build a task list of length 20
        for _ in range(repetition):
            task = create_task(input_map=grid_map, mode='free',
                               non_task_ep_list=None, task_list=None)
            if len(task_list) < 20:
                task_list.append(task)

            self.assertIsInstance(task, tuple)
            for loc in task:
                self.assertIsInstance(loc, tuple)
                self.assertEqual(0, grid_map[loc])

        '''avoid_non_task_rep'''
        for _ in range(repetition):
            task = create_task(input_map=grid_map, mode='avoid_non_task_rep',
                               non_task_ep_list=non_task_ep_list, task_list=None)

            for loc in task:
                self.assertEqual(0, grid_map[loc])
                self.assertNotIn(loc, non_task_ep_list)

        '''avoid_task_rep'''
        for _ in range(repetition):
            task = create_task(input_map=grid_map, mode='avoid_task_rep',
                               non_task_ep_list=None, task_list=task_list)

            self.assertNotIn(task, task_list)
            for loc in task:
                self.assertEqual(0, grid_map[loc])

        '''avoid_all'''
        for _ in range(repetition):
            task = create_task(input_map=grid_map, mode='avoid_all',
                               non_task_ep_list=non_task_ep_list,
                               task_list=task_list)

            self.assertNotIn(task, task_list)
            for loc in task:
                self.assertEqual(0, grid_map[loc])
                self.assertNotIn(loc, non_task_ep_list)


if __name__ == '__main__':
    unittest.main()
