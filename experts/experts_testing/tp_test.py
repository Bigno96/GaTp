import statistics
import timeit
import unittest
from pprint import pprint

from create_dataset.map_creator import create_random_grid_map
from experts.experts_testing.tp_agent_test import get_start_pos_non_tep_task_list
from experts.token_passing import tp
from utils.metrics import count_collision


class TpTest(unittest.TestCase):
    def test_tp(self):
        repetition = 1
        time_list = []
        collision_count_list = []
        makespan_list = []
        service_time_list = []
        timestep_runtime_list = []
        shape = (20, 40)
        density = 0.1
        agent_num = 50
        task_num = 500
        imm_task_split = 0
        new_task_per_timestep = 1
        step_between_insertion = 1

        start_pos_list = []
        task_list = []
        agent_schedule = {}

        for i in range(repetition):
            # map creation
            grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

            # non task endpoints
            # non task endpoints, task list
            start_pos_list, non_task_ep_list, task_list = get_start_pos_non_tep_task_list(input_map=grid_map,
                                                                                          agent_num=agent_num,
                                                                                          task_num=task_num)
            parking_spot_list = list(set(non_task_ep_list)-set(start_pos_list))

            # relaunch tp to get agent schedule and timer
            start_time = timeit.default_timer()

            agent_schedule, service_time, timestep_runtime = tp(input_map=grid_map,
                                                                start_pos_list=start_pos_list,
                                                                task_list=task_list,
                                                                parking_spot_list=parking_spot_list,
                                                                imm_task_split=imm_task_split,
                                                                new_task_per_insertion=new_task_per_timestep,
                                                                step_between_insertion=step_between_insertion)

            # write time
            time_diff = timeit.default_timer() - start_time
            time_list.append(time_diff)

            # collect conflicts
            coll_count, _ = count_collision(agent_schedule=agent_schedule)
            collision_count_list.append(coll_count)

            # collect makespan
            makespan_list.append(len(agent_schedule[0]))

            # collect timings
            service_time_list.append(service_time)
            timestep_runtime_list.append(timestep_runtime)

            self.assertIsInstance(agent_schedule, dict)
            length = len(agent_schedule[0])
            for schedule in agent_schedule.values():
                self.assertEqual(length, len(schedule))

            print(f'Solved scenario {i+1}/{repetition}')

        print(f'Agents starting positions: {start_pos_list}')
        print('Task List:')
        pprint(task_list)
        print('Resulting Schedule:')
        for schedule in agent_schedule.items():
            print(schedule)
        print(f'Average makespan: {statistics.mean(makespan_list)}')
        print(f'Average service time: {statistics.mean(service_time_list)}')
        print(f'Average timestep runtime: {statistics.mean(timestep_runtime_list)} s')
        print(f'Average TP execution time: {statistics.mean(time_list)} s')
        print(f'Number of instances with collisions: {sum(i > 0 for i in collision_count_list)} out of {repetition}')

        # coll_count, collision_time_list = count_collision(agent_schedule=agent_schedule)
        # pprint(good_map)
        # print(f'Collision detected: {coll_count}')
        # print(f'Collision times: {collision_time_list}')


if __name__ == '__main__':
    unittest.main()
