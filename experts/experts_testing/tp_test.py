import unittest

from pprint import pprint
import timeit

from create_dataset.map_creator import create_random_grid_map
from create_dataset.scenario_creator import create_starting_pos, create_task
from experts.token_passing import tp


# noinspection DuplicatedCode
class TpTest(unittest.TestCase):
    def test_tp(self):
        shape = (20, 20)
        density = 0.2
        agent_num = 5
        task_num = 20
        imm_task_split = 0.5
        new_task_per_timestep = 1

        # map creation
        grid_map = create_random_grid_map(map_shape=shape, map_density=density)

        # non task endpoints
        start_pos_list = create_starting_pos(input_map=grid_map, agent_num=agent_num,
                                             mode='random')
        non_task_ep_list = start_pos_list.copy()

        # task list
        task_list = []
        for _ in range(task_num):
            task_list.append(create_task(input_map=grid_map, mode='avoid_non_task_rep',
                                         non_task_ep_list=non_task_ep_list))

        start_time = timeit.default_timer()
        agent_schedule = tp(input_map=grid_map, task_list=task_list,
                            start_pos_list=start_pos_list, parking_spot=set(),
                            imm_task_split=imm_task_split,
                            new_task_per_timestep=new_task_per_timestep)
        time_diff = timeit.default_timer() - start_time

        print(f'Agents starting positions: {start_pos_list}')
        print('Task List:')
        pprint(task_list)
        print('Resulting Schedule:')

        length = len(agent_schedule[0])
        for schedule in agent_schedule.items():
            print(schedule)
            self.assertEqual(length, len(schedule[1]))

        self.assertIsInstance(agent_schedule, dict)

        print(f'TP execution time: {time_diff}')


if __name__ == '__main__':
    unittest.main()
