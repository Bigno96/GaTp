import unittest

from utils.metrics import count_collision
from testing.test_utils import get_grid_map_free_cell_token, find_start_goal


class MetricsTest(unittest.TestCase):

    def test_count_collision(self):
        shape = (20, 20)
        density = 0.2
        agent_num = 20

        # get map and 2 positions in the map
        grid_map, free_cell_list, _ = get_grid_map_free_cell_token(shape=shape, density=density,
                                                                   agent_num=agent_num,
                                                                   token_path_length=0)
        start, goal = find_start_goal(input_map=grid_map,
                                      free_cell_list=free_cell_list)

        '''1) single node collision'''
        # define a colliding agent schedule at time 0
        ag_sched = {0: [(start[0], start[1], 0)],
                    1: [(start[0], start[1], 0)]}

        coll_count, coll_list = count_collision(agent_schedule=ag_sched)

        self.assertEqual(coll_count, 1)
        self.assertIn(0, coll_list)

        '''2) multiple node collision'''
        # define a colliding agent schedule at time 0 and 2
        ag_sched = {0: [(start[0], start[1], 0), (goal[0], goal[1], 1), (start[0], start[1], 2)],
                    1: [(start[0], start[1], 0), (start[0], start[1], 1), (start[0], start[1], 2)]}

        coll_count, coll_list = count_collision(agent_schedule=ag_sched)

        self.assertEqual(coll_count, 2)
        self.assertIn(0, coll_list)
        self.assertIn(2, coll_list)

        '''3) single swap collision'''
        # define a colliding agent schedule at time 0-1
        ag_sched = {0: [(start[0], start[1], 0), (goal[0], goal[1], 1)],
                    1: [(goal[0], goal[1], 0), (start[0], start[1], 1)]}

        coll_count, coll_list = count_collision(agent_schedule=ag_sched)

        self.assertEqual(coll_count, 1)
        self.assertIn(0, coll_list)

        '''4) multiple swap collision'''
        # define a colliding agent schedule at time 0-1, and 2-3
        ag_sched = {0: [(start[0], start[1], 0), (goal[0], goal[1], 1), (goal[0], goal[1], 2), (start[0], start[1], 3)],
                    1: [(goal[0], goal[1], 0), (start[0], start[1], 1), (start[0], start[1], 2), (goal[0], goal[1], 3)]}

        coll_count, coll_list = count_collision(agent_schedule=ag_sched)

        self.assertEqual(coll_count, 2)
        self.assertIn(0, coll_list)
        self.assertIn(2, coll_list)

        '''5) multiple combined collision'''
        # node collision at time 0 and 3, swap collision at time 1-2
        ag_sched = {1: [(start[0], start[1], 0), (goal[0], goal[1], 1), (start[0], start[1], 2),
                        (start[0], start[1], 3)],
                    2: [(start[0], start[1], 0), (start[0], start[1], 1), (goal[0], goal[1], 2),
                        (start[0], start[1], 3)]}

        coll_count, coll_list = count_collision(agent_schedule=ag_sched)

        self.assertEqual(coll_count, 3)
        self.assertIn(0, coll_list)
        self.assertIn(1, coll_list)
        self.assertIn(3, coll_list)

        '''6) schedule length 1 with no collision'''
        ag_sched = {0: [(start[0], start[1], 0)],
                    1: [(goal[0], goal[1], 0)]}

        coll_count, coll_list = count_collision(agent_schedule=ag_sched)

        self.assertEqual(coll_count, 0)
        self.assertFalse(coll_list)

        '''7) schedule length 2 with no collision'''
        ag_sched = {1: [(start[0], start[1], 0), (start[0], start[1], 1)],
                    2: [(goal[0], goal[1], 0), (goal[0], goal[1], 1)]}

        coll_count, coll_list = count_collision(agent_schedule=ag_sched)

        self.assertEqual(coll_count, 0)
        self.assertFalse(coll_list)


if __name__ == '__main__':
    unittest.main()
