import random
import unittest
from collections import deque
from pprint import pprint

import numpy as np

from create_dataset.map_creator import create_random_grid_map
from create_dataset.scenario_creator import create_task, create_starting_pos
from experts.tp_agent import TpAgent
from utils.expert_utils import preprocess_heuristics, compute_manhattan_heuristic


# noinspection DuplicatedCode
class TpAgentTest(unittest.TestCase):

    def test_preprocess_heuristics(self):
        # map creation
        shape = (20, 20)
        density = 0.2
        agent_num = 5
        task_num = 10
        grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

        # non task endpoints
        start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_num,
                                             mode='random')
        non_task_ep_list = start_pos_list.copy()

        # task list
        task_list = []
        for _ in range(task_num):
            task_list.append(create_task(input_map=grid_map, mode='avoid_non_task_rep',
                                         non_task_ep_list=non_task_ep_list))
        # check task list validity
        for pickup, delivery in task_list:
            self.assertNotIn(pickup, start_pos_list)
            self.assertNotIn(delivery, start_pos_list)

        # heuristic collection
        h_coll = preprocess_heuristics(input_map=grid_map, task_list=task_list,
                                       non_task_ep_list=non_task_ep_list)

        '''test heuristic collection '''
        self.assertIsInstance(h_coll, dict)
        for k, v in h_coll.items():
            self.assertIsInstance(v, np.ndarray)
            self.assertTrue((v == compute_manhattan_heuristic(grid_map, k)).all())

    def test_move_one_step(self):
        repetition = 10000
        path_len = 10
        shape = (20, 20)

        for _ in range(repetition):
            path = [(random.choice(range(shape[0])), random.choice(range(shape[1])), t)
                    for t in range(path_len)]

            agent = TpAgent(name='ag', input_map=None,
                            start_pos=path[0][:-1], h_coll=None)
            agent.path = deque(path)

            # move along the random path
            self.assertEqual(agent.pos, path[0][:-1])
            for x, y, t in path:
                agent.move_one_step()
                self.assertEqual(agent.pos, (x, y))

            # check stays in place after the path
            agent.move_one_step()
            self.assertEqual(agent.pos, path[-1][:-1])

    def test_find_resting_pos(self):
        repetition = 100
        for i in range(repetition):
            # map creation
            shape = (20, 20)
            density = 0.2
            agent_num = 5
            task_num = 10
            grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

            # non task endpoints
            start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_num,
                                                 mode='random')
            non_task_ep_list = start_pos_list.copy()

            # task list
            task_list = []
            for _ in range(task_num):
                task_list.append(create_task(input_map=grid_map, mode='avoid_non_task_rep',
                                             non_task_ep_list=start_pos_list))
            task_delivery_list = [delivery for
                                  _, delivery in task_list]
            # check task list validity
            for pickup, delivery in task_list:
                self.assertNotIn(pickup, start_pos_list)
                self.assertNotIn(delivery, start_pos_list)

            # heuristic collection
            h_coll = preprocess_heuristics(input_map=grid_map, task_list=task_list,
                                           non_task_ep_list=non_task_ep_list)

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
            token_start_pos_list = [path[0][:-1]
                                    for path in token.values()]
            token_ep_list = [path[-1][:-1]
                             for path in token.values()]
            # check token validity
            for x, y, t in token_step_list:
                self.assertEqual(grid_map[(x, y)], 0)

            # set Tp Agent instance
            agent = TpAgent(name='ag', input_map=grid_map,
                            start_pos=(0, 0), h_coll=h_coll)    # start pos here doesn't matter

            '''testing find_resting_pos'''
            # test an agent which has just finished a task and needs to find resting position
            # start from position outside non-task endpoints or first positions in token's paths
            # start_pos can be in task list, since task can have repeating positions
            for _ in range(repetition):
                start_pos = random.choice(list(set(free_cell_list) - set(non_task_ep_list) - set(token_start_pos_list)))

                agent.pos = start_pos
                agent.find_resting_pos(token=token, task_list=task_list, non_task_ep_list=non_task_ep_list,
                                       sys_timestep=0)

                # check validity of self.path -> endpoint not in tasks delivery location (pickup allowed),
                #                                nor in token endpoints
                if len(agent.path) > 1:
                    self.assertNotIn(agent.path[-1][:-1], token_ep_list)
                    self.assertNotIn(agent.path[-1][:-1], task_delivery_list)
                # if agent is standing still, due to not well-formed MAPD instances
                # or to agent already standing on a correct endpoint
                else:
                    self.assertEqual((agent.path[0][0], agent.path[0][1], agent.path[0][2]),
                                     (start_pos[0], start_pos[1], 0))

    def test_receive_token(self):
        repetition = 1000
        printed = False
        for idx in range(repetition):
            # map creation
            shape = (20, 20)
            density = 0.2
            agent_num = 5
            task_num = 10
            grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

            # non task endpoints
            start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_num,
                                                 mode='random')
            non_task_ep_list = start_pos_list.copy()

            # task list
            task_list = []
            for _ in range(task_num):
                task_list.append(create_task(input_map=grid_map, mode='avoid_non_task_rep',
                                             non_task_ep_list=non_task_ep_list))
            task_delivery_list = [delivery for
                                  _, delivery in task_list]
            task_pickup_list = [pickup for
                                pickup, _ in task_list]
            # check task list validity
            for pickup, delivery in task_list:
                self.assertNotIn(pickup, start_pos_list)
                self.assertNotIn(delivery, start_pos_list)

            # heuristic collection
            h_coll = preprocess_heuristics(input_map=grid_map, task_list=task_list,
                                           non_task_ep_list=non_task_ep_list)

            # get free cell positions
            where_res = np.nonzero(grid_map == 0)
            free_cell_list = list(zip(where_res[0], where_res[1]))

            # token
            token = {1: deque([(x, y, t)
                               for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))]),
                     2: deque([(x, y, t)
                               for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))]),
                     3: deque([(x, y, t)
                               for t, (x, y) in enumerate(random.sample(population=free_cell_list, k=5))])}
            x, y = random.choice(seq=free_cell_list)
            token['stands_still'] = deque([(x, y, 0)])  # one agent stands still

            token_step_list = [step
                               for path in token.values()
                               for step in path]
            token_start_pos_list = [path[0][:-1]
                                    for path in token.values()]
            token_ep_list = [path[-1][:-1]
                             for path in token.values()]
            # check token validity
            for x, y, t in token_step_list:
                self.assertEqual(grid_map[(x, y)], 0)

            # set Tp Agent instance
            agent = TpAgent(name='ag', input_map=grid_map,
                            start_pos=start_pos_list[0], h_coll=h_coll)
            token['ag'] = agent.path

            '''test TpAgent receive token'''
            if not printed:
                # check token
                print('Token before agent exec:')
                pprint(token)
                # check task list
                print('\nTask list before agent exec:')
                pprint(task_list)

            agent.receive_token(token=token, task_list=task_list, non_task_ep_list=non_task_ep_list,
                                sys_timestep=0)

            if not printed:
                print('\nToken after agent exec:')
                pprint(token)
                # check task list
                print('\nTask list after agent exec:')
                pprint(task_list)

                printed = True

            '''limit testing'''
            # no tasks available -> move accordingly

            # 1) agents is in a non-conflicting position, so it stands still
            del token[agent.name]
            start_pos = random.choice(list(set(free_cell_list)
                                           - set(task_delivery_list)    # not in delivery locs
                                           - set(token_ep_list)      # not in token paths endpoints
                                           - set(token_start_pos_list)))    # not in another agent current pos
            agent.pos = start_pos
            agent.receive_token(token=token, task_list=[], non_task_ep_list=non_task_ep_list, sys_timestep=0)

            self.assertEqual(start_pos, agent.path[0][:-1])     # stand still
            self.assertTrue(agent.is_free)

            # 2) conflicting position, reposition into an available endpoint
            del token[agent.name]
            # select start pos in delivery locs or token paths endpoints
            # can't select agent 'stands_still' endpoint since it's also its start position
            # this is discarded only for this specific test
            # during execution, this situation is never occurring (node conflicts aren't allowed)
            token_ep_list.remove(token['stands_still'][-1][:-1])
            start_pos = random.choice(list(set(task_delivery_list + token_ep_list)))

            agent.pos = start_pos
            # another agent going into start pos ('occupy' that task)
            token['stands_still'].append((start_pos[0], start_pos[1], 1))
            agent.receive_token(token=token, task_list=[(start_pos, start_pos)],   # set one task in the start agent pos
                                non_task_ep_list=non_task_ep_list, sys_timestep=0)

            # build available positions list to check execution
            token_ep_list.append(token['stands_still'][-1][:-1])
            avail_pos = non_task_ep_list + task_pickup_list
            avail_pos = set(avail_pos) - set(task_delivery_list) - set(token_ep_list)

            if len(agent.path) > 1:
                self.assertIn(agent.path[-1][:-1], avail_pos)  # moves to an available endpoint
                self.assertTrue(agent.is_free)
            else:
                self.assertEqual(agent.path[-1][:-1], start_pos)    # can't find a path to move, stand still
                self.assertTrue(agent.is_free)


if __name__ == '__main__':
    unittest.main()
