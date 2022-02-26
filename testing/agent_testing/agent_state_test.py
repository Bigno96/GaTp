import unittest
import random
import numpy as np
import utils.agent_state as ag_st

from easydict import EasyDict


class AgentStateTest(unittest.TestCase):

    @staticmethod
    def init_ag_state():
        # init agent state instance
        agent_num = 10
        FOV = 9
        return ag_st.AgentState(EasyDict({'agent_number': agent_num,
                                             'FOV': FOV}))

    @staticmethod
    def get_obstacle_map():
        # get obstacle map with specified shape and obstacle density
        H = 20
        W = 20
        density = 0.1

        obs_pos_list = np.unravel_index(random.sample(range(int(H * W)), k=int(density * H * W)), shape=(H, W))
        obs_pos_list = list(zip(obs_pos_list[0], obs_pos_list[1]))
        obs_map = np.zeros(shape=(H, W), dtype=np.int8)
        for p in obs_pos_list:
            obs_map[p] = 1

        return obs_map

    @staticmethod
    def get_agent_goal_pos_lists(num_agents, input_map):
        # get list of agent pos and goal pos
        # list -> np.ndarray, shape=(num_agent, 2)
        free_cell_list = np.nonzero(input_map == 0)
        free_cell_list = list(zip(free_cell_list[0], free_cell_list[1]))

        agent_pos_list = np.array(random.sample(free_cell_list, k=num_agents), dtype=np.int8)
        goal_pos_list = np.array(random.sample(free_cell_list, k=num_agents), dtype=np.int8)

        return agent_pos_list, goal_pos_list

    def test_set_obstacle_map(self):
        ag_state = self.init_ag_state()
        obs_map = self.get_obstacle_map()

        ag_state.set_obstacle_map(obs_map)

        # check normal map equality
        self.assertTrue(np.array_equal(obs_map.shape, ag_state.obstacle_map.shape))

        # check padded size
        H, W = obs_map.shape
        H_pad = H + int(2 * ag_state.FOV_width)
        W_pad = W + int(2 * ag_state.FOV_width)
        self.assertEqual((H_pad, W_pad), ag_state.obstacle_map_pad.shape)
        # check padded obstacles
        # all the outside is padded with 1s
        self.assertEqual(np.count_nonzero(ag_state.obstacle_map_pad),
                         int(np.count_nonzero(obs_map)
                             + 2 * ag_state.FOV_width * obs_map.shape[0]        # vertical sides
                             + 2 * ag_state.FOV_width * obs_map.shape[1]        # horizontal sides
                             + 4 * ag_state.FOV_width * ag_state.FOV_width))    # corners

    def test_get_agent_pos_map(self):
        ag_state = self.init_ag_state()
        obs_map = self.get_obstacle_map()
        agent_pos_list, _ = self.get_agent_goal_pos_lists(num_agents=ag_state.agent_number,
                                                          input_map=obs_map)

        # obstacle map not set
        self.assertRaises(AssertionError,
                          lambda: ag_state.get_agent_pos_map(agent_pos_list=agent_pos_list))

        ag_state.set_obstacle_map(input_map=obs_map)
        agent_pos_map_pad = ag_state.get_agent_pos_map(agent_pos_list=agent_pos_list)

        # check size
        H, W = obs_map.shape
        H_pad = H + int(2 * ag_state.FOV_width)
        W_pad = W + int(2 * ag_state.FOV_width)
        self.assertEqual((H_pad, W_pad), agent_pos_map_pad.shape)

        # check number of 1s
        self.assertEqual(np.count_nonzero(agent_pos_map_pad), ag_state.agent_number)

        # check positions
        # add FOV width to original pos coordinates
        for p in agent_pos_list:
            x, y = p
            x_pad = x + ag_state.FOV_width
            y_pad = y + ag_state.FOV_width
            self.assertEqual(agent_pos_map_pad[(x_pad, y_pad)], 1)

    def test_project_goal(self):
        ag_state = self.init_ag_state()
        obs_map = self.get_obstacle_map()

        agent_x, agent_y = (13, 9)
        # slicing exclude end of the interval
        # skewing by FOV_width since agent_pos is based on a non-padded map, while the FOV is cut out of a padded one
        FOV_x = [agent_x, agent_x + int(2*ag_state.FOV_width) + 1]      # [9, 17+1] non-padded coord
        FOV_y = [agent_y, agent_y + int(2*ag_state.FOV_width) + 1]      # [5, 13+1] non-padded coord
        # outside FOV
        goal_x, goal_y = (18, 17)     # non-padded coord

        # build goal map, cut FOV and pad a border around
        goal_pos_map = np.zeros_like(obs_map, dtype=np.int8)  # full map
        goal_pos_map[goal_x][goal_y] = 1  # set to 1 the agent goal
        goal_pos_map_pad = np.pad(goal_pos_map, pad_width=ag_state.FOV_width,  # add border around the full map
                                  mode='constant', constant_values=0)
        goal_pos_map_FOV = goal_pos_map_pad[FOV_x[0]:FOV_x[1], FOV_y[0]:FOV_y[1]]  # cut the FOV out

        projected_map = ag_state.project_goal(goal_map_FOV=goal_pos_map_FOV,
                                              agent_pos=(agent_x, agent_y),
                                              goal_pos=(goal_x, goal_y))

        # check size
        L = ag_state.FOV + int(2*ag_state.border)
        self.assertEqual(projected_map.shape, (L, L))

        # only 1 goal
        self.assertEqual(np.count_nonzero(projected_map), 1)

    def test_build_input_state(self):
        repetition = 1000

        for i in range(repetition):
            ag_state = self.init_ag_state()
            obs_map = self.get_obstacle_map()
            agent_pos_list, goal_pos_list = self.get_agent_goal_pos_lists(num_agents=ag_state.agent_number,
                                                                          input_map=obs_map)
            ag_state.set_obstacle_map(input_map=obs_map)
            agent_pos_map_pad = ag_state.get_agent_pos_map(agent_pos_list)
            id_agent = random.choice(range(ag_state.agent_number))

            # agent state summary
            load_state = (goal_pos_list, agent_pos_list, agent_pos_map_pad, id_agent)

            # input_state -> 3 input channels (maps): obstacle map, goal map, agents pos map
            # all map are FOV cut and padded with border
            input_state_curr_agent, local_objective = ag_state.build_agent_input_state(load_state=load_state)
            obs_map_FOV_pad, goal_map_FOV_pad, agent_map_FOV_pad = input_state_curr_agent

            '''check sizes'''
            L = ag_state.FOV + int(2*ag_state.border)
            self.assertEqual(obs_map_FOV_pad.shape, (L, L))
            self.assertEqual(goal_map_FOV_pad.shape, (L, L))
            self.assertEqual(agent_map_FOV_pad.shape, (L, L))

            '''check goal channel'''
            self.assertEqual(np.count_nonzero(goal_map_FOV_pad), 1)     # only 1 goal
            # set up variables, all in original map coord reference
            goal_pos_x, goal_pos_y = goal_pos_list[id_agent]
            ag_pos_x, ag_pos_y = agent_pos_list[id_agent]
            # fov without border
            fov_x = range(ag_pos_x - ag_state.FOV_width, ag_pos_x + ag_state.FOV_width + 1)
            fov_y = range(ag_pos_y - ag_state.FOV_width, ag_pos_y + ag_state.FOV_width + 1)
            goal_inside_FOV = goal_pos_x in fov_x and goal_pos_y in fov_y       # goal is inside agent FOV

            # goal pos in padded FOV map
            goal_pos_fov_pad = np.nonzero(goal_map_FOV_pad)
            goal_pos_fov_pad = tuple(zip(goal_pos_fov_pad[0], goal_pos_fov_pad[1]))[0]      # unpack

            # goal inside agent FOV -> not in the border
            if goal_inside_FOV:
                self.assertTrue(0 < goal_pos_fov_pad[0] < ag_state.FOV+1)       # inside border
                self.assertTrue(0 < goal_pos_fov_pad[1] < ag_state.FOV+1)       # inside border
            # goal outside agent FOV -> in the border
            else:
                self.assertTrue(0 == goal_pos_fov_pad[0]
                                or goal_pos_fov_pad[0] == ag_state.FOV+1
                                or 0 == goal_pos_fov_pad[1]
                                or goal_pos_fov_pad[1] == ag_state.FOV+1)    # on the border

            '''check agent channel'''
            # pos of agent in the coord frame of padded FOV
            ag_pos_FOV_pad = (ag_state.center_x, ag_state.center_y)
            # agent himself always = 1
            self.assertEqual(agent_map_FOV_pad[ag_pos_FOV_pad], 1)

            # get how many other agents should be inside FOV
            ag_fov_count = 0
            for id_ag in range(ag_state.agent_number):
                # if agent is inside fov
                if agent_pos_list[id_ag][0] in fov_x and agent_pos_list[id_ag][1] in fov_y:
                    ag_fov_count += 1       # always counting agent himself

            self.assertEqual(ag_fov_count, np.count_nonzero(agent_map_FOV_pad))

            '''check obstacle channel'''
            # skew fov to use padded coord
            fov_x_pad = [ag_pos_x, ag_pos_x + int(2*ag_state.FOV_width) + 1]
            fov_y_pad = [ag_pos_y, ag_pos_y + int(2*ag_state.FOV_width) + 1]
            # get obstacle padded full map
            obs_map_pad = ag_state.obstacle_map_pad

            # check same number of obstacles
            self.assertEqual(np.count_nonzero(obs_map_FOV_pad),
                             np.count_nonzero(obs_map_pad[fov_x_pad[0]:fov_x_pad[1],
                                                          fov_y_pad[0]:fov_y_pad[1]]))

            '''check local objective'''
            if goal_inside_FOV:
                # same as original goal position
                self.assertEqual(tuple(local_objective), (goal_pos_x, goal_pos_y))
            else:
                # on the border of the FOV, skewed by self.FOV_width + self.border
                loc_x, loc_y = (local_objective[0], local_objective[1])
                # consider border padding
                self.assertTrue(loc_x == fov_x[0]-ag_state.border
                                or loc_x == fov_x[-1]+ag_state.border
                                or loc_y == fov_y[0]-ag_state.border
                                or loc_y == fov_y[-1]+ag_state.border)

    def test_get_input_state(self):
        # validity of size and values already checked in test build input state
        ag_state = self.init_ag_state()
        obs_map = self.get_obstacle_map()
        agent_pos_list, goal_pos_list = self.get_agent_goal_pos_lists(num_agents=ag_state.agent_number,
                                                                      input_map=obs_map)
        ag_state.set_obstacle_map(input_map=obs_map)

        input_state = ag_state.get_input_state(goal_pos_list=goal_pos_list,
                                                agent_pos_list=agent_pos_list)

        # check execution, returned type and size of tensor
        self.assertIsInstance(input_state, np.ndarray)
        channel_num = 3
        # shape = (num_agents, channels, FOV_H+2*border, FOV_W+2*border)
        self.assertEqual((ag_state.agent_number, channel_num, ag_state.H, ag_state.W),
                         input_state.shape)

        # check local objective list
        local_objective_list = ag_state.get_local_obj_list()

        self.assertEqual((ag_state.agent_number, 2), local_objective_list.shape)

    def test_get_sequence_input_state(self):
        # validity of size and values already checked in test build input state
        ag_state = self.init_ag_state()
        obs_map = self.get_obstacle_map()
        agent_pos_list, goal_pos_list = self.get_agent_goal_pos_lists(num_agents=ag_state.agent_number,
                                                                      input_map=obs_map)
        ag_state.set_obstacle_map(input_map=obs_map)
        makespan = 5

        # set up agent schedule, shape = (makespan, num_agents, 2)
        ag_schedule = np.tile(agent_pos_list, reps=(makespan, 1, 1))
        # set up goal schedule, shape = (makespan, num_agents, 2)
        goal_schedule = np.tile(goal_pos_list, reps=(makespan, 1, 1))

        input_state = ag_state.get_sequence_input_state(goal_pos_schedule=goal_schedule,
                                                         agent_pos_schedule=ag_schedule,
                                                         makespan=makespan)

        # check execution, returned type and size of tensor
        self.assertIsInstance(input_state, np.ndarray)
        channel_num = 3
        # shape = (makespan, num_agents, channels, FOV_H+2*border, FOV_W+2*border)
        self.assertEqual((makespan, ag_state.agent_number, channel_num, ag_state.H, ag_state.W),
                         input_state.shape)


if __name__ == '__main__':
    unittest.main()
