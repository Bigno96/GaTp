import random
import unittest
from collections import deque
from copy import deepcopy
from pprint import pprint

import numpy as np

from create_dataset.map_creator import create_random_grid_map
from create_dataset.scenario_creator import create_task, create_starting_pos
from experts.experts_testing.a_star_test import get_grid_map_free_cell_token
from experts.tp_agent import TpAgent
from utils.expert_utils import preprocess_heuristics, compute_manhattan_heuristic
from utils.metrics import count_collision


# noinspection DuplicatedCode
class TpAgentTest(unittest.TestCase):

    def test_preprocess_heuristics(self):
        shape = (20, 20)
        density = 0.2
        agent_num = 5
        task_num = 10
        # input map
        grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)
        # non task endpoints, task list
        _, non_task_ep_list, task_list = get_start_pos_non_tep_task_list(input_map=grid_map,
                                                                         agent_num=agent_num, task_num=task_num)
        # heuristic collection
        h_coll = preprocess_heuristics(input_map=grid_map, task_list=task_list,
                                       non_task_ep_list=non_task_ep_list)

        '''test heuristic collection '''
        self.assertIsInstance(h_coll, dict)
        for k, v in h_coll.items():
            self.assertIsInstance(v, np.ndarray)
            self.assertTrue((v == compute_manhattan_heuristic(grid_map, k)).all())

    def test_move_one_step(self):
        path_len = 10
        shape = (20, 20)
        # random path to move along
        path = [(random.choice(range(shape[0])), random.choice(range(shape[1])), t)
                for t in range(path_len)]
        # create agent and give him the path
        agent = TpAgent(name='ag', input_map=None,
                        start_pos=path[0][:-1], h_coll=None)
        agent.path = deque(path)
        agent.is_free = False

        '''move along the path'''
        self.assertEqual(agent.pos, path[0][:-1])       # check start pos
        # check movement consistency
        for x, y, t in path:
            self.assertFalse(agent.is_free)
            agent.move_one_step()
            self.assertEqual(agent.pos, (x, y))
        # last step, agent became free
        self.assertTrue(agent.is_free)
        # check stays in place after the path, updating its timestep
        self.assertEqual(agent.pos, path[-1][:-1])
        self.assertEqual(path_len, agent.path[-1][-1])

    def test_find_resting_pos(self):
        repetition = 1000
        shape = (20, 20)
        density = 0.2
        agent_num = 5
        task_num = 10
        tok_path_length = 15

        for idx in range(repetition):
            # get map, free cell list and token
            grid_map, free_cell_list, token = get_grid_map_free_cell_token(shape=shape, density=density,
                                                                           agent_num=agent_num-1,
                                                                           token_path_length=tok_path_length)
            # no standing still here, since no collision shielding
            del(token['stands_still'])

            # non task endpoints, task list
            _, non_task_ep_list, task_list = get_start_pos_non_tep_task_list(input_map=grid_map,
                                                                             agent_num=agent_num, task_num=task_num)
            task_delivery_list = [delivery for
                                  _, delivery in task_list]

            # heuristic collection
            h_coll = preprocess_heuristics(input_map=grid_map, task_list=task_list,
                                           non_task_ep_list=non_task_ep_list)

            # get information from token
            _, token_start_pos_list, token_ep_list = get_tok_posl_startl_epl(token=token)

            # set Tp Agent instance
            agent = TpAgent(name='ag', input_map=grid_map,
                            start_pos=(0, 0), h_coll=h_coll)    # start pos here doesn't matter

            '''testing find_resting_pos'''
            # test an agent which needs to find resting position
            # start from position outside non-task endpoints (cause otherwise find_resting_pos is not called)
            # exclude also first positions in token's paths, impossible to be there (conflicts)
            # no other limitations on starting pos
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
            else:
                self.assertEqual((agent.path[0][0], agent.path[0][1], agent.path[0][2]),
                                 (start_pos[0], start_pos[1], 0))
            # path has been added to the token and agent is free
            self.assertEqual(token['ag'], agent.path)
            self.assertTrue(agent.is_free)

            agent_schedule = build_ag_schedule(token=token, agent_name='ag')
            collision_count, l_ = count_collision(agent_schedule=agent_schedule)

            # no collision provoked by agent
            self.assertFalse(collision_count)

    def test_receive_token(self):
        repetition = 1000
        printed = False
        shape = (20, 20)
        density = 0.2
        agent_num = 5
        task_num = 10

        for idx in range(repetition):
            grid_map, free_cell_list, token = get_grid_map_free_cell_token(shape=shape, density=density,
                                                                           agent_num=agent_num-1,
                                                                           token_path_length=15)
            del(token['stands_still'])      # causing collision due to absence of collision shielding

            # starting positions, non task endpoints, task list
            start_pos_list, non_task_ep_list, task_list = get_start_pos_non_tep_task_list(input_map=grid_map,
                                                                                          agent_num=agent_num,
                                                                                          task_num=task_num)
            task_delivery_list = [delivery for
                                  _, delivery in task_list]

            # heuristic collection
            h_coll = preprocess_heuristics(input_map=grid_map, task_list=task_list,
                                           non_task_ep_list=non_task_ep_list)

            # get information from token
            token_pos_list, token_start_pos_list, token_ep_list = get_tok_posl_startl_epl(token=token)

            # set Tp Agent instance, not in a position in the token
            start_pos = random.choice(list(set(free_cell_list) - set(token_pos_list)))
            # why? if it stands still since no task assigned, no collision shielding and might collide
            agent = TpAgent(name='ag', input_map=grid_map,
                            start_pos=start_pos, h_coll=h_coll)
            agent.is_free = True
            token[agent.name] = agent.path

            '''test TpAgent receive token'''
            if not printed:
                # check token
                print('Token before agent exec:')
                for p in token.items():
                    print(p)
                # check task list
                print('\nTask list before agent exec:')
                pprint(task_list)

            task_num = len(task_list)
            agent.receive_token(token=token, task_list=task_list, non_task_ep_list=non_task_ep_list,
                                sys_timestep=0)

            self.assertIsNotNone(token[agent.name])
            if agent.is_free:
                self.assertEqual(task_num, len(task_list))      # no task assigned
            else:
                self.assertEqual(task_num-1, len(task_list))    # one task assigned

            agent_schedule = build_ag_schedule(token=token, agent_name=agent.name)
            collision_count, l_ = count_collision(agent_schedule=agent_schedule)

            # no collision
            self.assertFalse(collision_count)

            if not printed:
                print('\nToken after agent exec:')
                for p in token.items():
                    print(p)
                # check task list
                print('\nTask list after agent exec:')
                pprint(task_list)

                printed = True

            '''limit testing'''
            # no tasks available -> move accordingly

            '''a) agents is in a non-conflicting position, so it stands still'''
            start_pos = random.choice(list(set(free_cell_list)
                                           - set(task_delivery_list)    # not in delivery locs
                                           - set(token_ep_list)      # not in token paths endpoints
                                           - set(token_start_pos_list)))    # not in another agent current pos
            agent.pos = start_pos
            agent.is_free = True
            agent.path = deque([(start_pos[0], start_pos[1], 0)])
            token[agent.name] = agent.path
            # no task to assign himself
            agent.receive_token(token=token, task_list=[], non_task_ep_list=non_task_ep_list, sys_timestep=0)

            self.assertEqual(start_pos, agent.path[0][:-1])     # stand still
            self.assertTrue(agent.is_free)
            self.assertIsNotNone(token[agent.name])

            '''b) conflicting position, reposition into an available endpoint'''
            '''B1 -> task unavailable due to delivery position blocked'''
            # select start pos in delivery locs
            pickup, start_pos = random.choice(list(set(task_list)))

            agent.pos = start_pos
            agent.is_free = True
            agent.path = deque([(start_pos[0], start_pos[1], 0)])
            token[agent.name] = agent.path
            # another agent going into start pos == delivery pos of the only task available
            # -> 'ag' can't assign to the only task available
            # -> 'ag' is in a delivery spot
            # -> 'ag' will move with 'find_resting_pos'
            token['stands_still'] = [(start_pos[0], start_pos[1], 1)]
            agent.receive_token(token=token, task_list=[(pickup, start_pos)],
                                non_task_ep_list=non_task_ep_list, sys_timestep=0)

            # build available positions list to check execution
            avail_pos = non_task_ep_list + [pickup]
            avail_pos = set(avail_pos) - {start_pos} - set(token_ep_list)

            self.assertIsNotNone(token[agent.name])
            if len(agent.path) > 1:
                self.assertIn(agent.path[-1][:-1], avail_pos)  # moves to an available endpoint
                self.assertTrue(agent.is_free)
            else:
                self.assertEqual(agent.path[-1][:-1], start_pos)    # can't find a path to move, stand still
                self.assertTrue(agent.is_free)

            '''B2 -> task unavailable due to pickup position blocked'''
            # select start pos in delivery locs
            agent.pos = start_pos
            agent.is_free = True
            agent.path = deque([(start_pos[0], start_pos[1], 0)])
            token[agent.name] = agent.path
            # another agent going into start pos == pickup pos of the only task available
            # -> 'ag' can't assign to the only task available
            # -> 'ag' is in a delivery spot
            # -> 'ag' will move with 'find_resting_pos'
            token['stands_still'] = [(pickup[0], pickup[1], 2)]
            agent.receive_token(token=token, task_list=[(pickup, start_pos)],
                                non_task_ep_list=non_task_ep_list, sys_timestep=0)

            # build available positions list to check execution
            avail_pos = non_task_ep_list + [pickup]
            avail_pos = set(avail_pos) - {start_pos} - set(token_ep_list)

            self.assertIsNotNone(token[agent.name])
            if len(agent.path) > 1:
                self.assertIn(agent.path[-1][:-1], avail_pos)  # moves to an available endpoint
                self.assertTrue(agent.is_free)
            else:
                self.assertEqual(agent.path[-1][:-1], start_pos)  # can't find a path to move, stand still
                self.assertTrue(agent.is_free)

    def test_collision_shielding(self):
        repetition = 1000
        shape = (20, 20)
        density = 0.2
        starting_t_range = 20

        for idx in range(repetition):
            grid_map, free_cell_list, _ = get_grid_map_free_cell_token(shape=shape, density=density,
                                                                       agent_num=0,
                                                                       token_path_length=0)
            token = {}

            '''Collision shielding activation case'''
            # agent 1 is standing still, after it has finished its path
            # agent 2 is moving along its path -> moves into ag1 endpoint AFTER ag1 has reached it
            # agent 1 uses collision shielding to move away
            # if ag2 moves into ag1 endpoint BEFORE ag1, 2 cases:
            #       a) it moves away the next timestep -> no conflicts
            #       b) it stands there -> roles are flipped, but same scenario as above

            ''' 1) ag2 comes into ag1 the next timestep after ag1 finishes its path'''
            start_pos_ag1, start_pos_ag2 = random.sample(population=free_cell_list, k=2)
            agent1 = TpAgent(name='ag1', input_map=grid_map, start_pos=start_pos_ag1, h_coll=None)  # h coll doesnt matter
            agent2 = TpAgent(name='ag2', input_map=grid_map, start_pos=start_pos_ag2, h_coll=None)  # h coll doesnt matter

            starting_t = random.choice(range(starting_t_range))
            # ag1 standing still
            agent1.path = deque([(start_pos_ag1[0], start_pos_ag1[1], starting_t)])
            token[agent1.name] = agent1.path
            # ag2 coming into ag1.pos 'after' him
            agent2.path = deque([(start_pos_ag2[0], start_pos_ag2[1], starting_t),
                                 (start_pos_ag1[0], start_pos_ag1[1], starting_t+1)])
            token[agent2.name] = agent2.path

            # apply to both collision shielding
            agent1.collision_shielding(token=token, sys_timestep=starting_t)
            agent2.collision_shielding(token=token, sys_timestep=starting_t)

            # ag2 doesn't utilize coll shield
            self.assertEqual(agent2.path, deque([(start_pos_ag2[0], start_pos_ag2[1], starting_t),
                                                 (start_pos_ag1[0], start_pos_ag1[1], starting_t+1)]))

            # ag1 utilizes coll shield
            if len(agent1.path) > 1:
                self.assertFalse(agent1.is_free)
                self.assertNotEqual(agent1.path[1], agent2.path[1])
            else:
                # verify no available cell around him
                d_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                c = start_pos_ag1
                cell_around = [(c[0]+d[0], c[1]+d[1])
                               for d in d_list]
                cell_around = [cell
                               for cell in cell_around
                               if (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1])
                               and grid_map[cell] == 0
                               and cell != agent2.path[1][:-1]
                               and cell != agent2.path[0][:-1]]

                self.assertFalse(cell_around)       # no cell around
                self.assertTrue(agent1.is_free)


def get_start_pos_non_tep_task_list(input_map, agent_num, task_num):
    # non task endpoints
    start_pos_list = create_starting_pos(input_map=input_map, agent_num=agent_num,
                                         mode='random')
    parking_spot = []
    non_task_ep_list = start_pos_list + parking_spot

    # task list
    task_list = []
    for _ in range(task_num):
        task_list.append(create_task(input_map=input_map, mode='avoid_non_task_rep',
                                     non_task_ep_list=non_task_ep_list))

    return start_pos_list, non_task_ep_list, task_list


def get_tok_posl_startl_epl(token):
    token_pos_list = [(x, y)
                      for path in token.values()
                      for x, y, t in path]
    token_start_pos_list = [path[0][:-1]
                            for path in token.values()]
    token_ep_list = [path[-1][:-1]
                     for path in token.values()]

    return token_pos_list, token_start_pos_list, token_ep_list


def build_ag_schedule(token, agent_name):
    agent_schedule = deepcopy(token)
    # make all the paths in the token the same length
    max_len = max([len(path)
                   for ag, path in token.items()
                   if ag != agent_name
                   ])
    agent_len = len(token[agent_name])

    max_len = min(max_len, agent_len)
    for ag, path in token.items():
        diff = max_len - len(path)
        # path was shorter
        if diff >= 0:
            for i in range(diff):
                agent_schedule[ag].append((path[-1][0], path[-1][1], path[-1][2]+i+1))
        else:
            # path might be a deque
            agent_schedule[ag] = list(path)[:agent_len].copy()

    return agent_schedule


if __name__ == '__main__':
    unittest.main()
