import statistics
import timeit
import unittest
from multiprocessing import Process
from pprint import pprint

from create_dataset.map_creator import create_random_grid_map
from experts.experts_testing.tp_agent_test import get_start_pos_non_tep_task_list
from experts.token_passing import tp
from utils.metrics import count_collision


# noinspection DuplicatedCode,PyUnboundLocalVariable
class TpTest(unittest.TestCase):
    def test_tp(self):
        repetition = 100
        time_list = []
        collision_count_list = []
        makespan_list = []
        bad_mapd_inst_count = 0
        shape = (20, 20)
        density = 0.2
        agent_num = 10
        task_num = 50
        imm_task_split = 0.5
        new_task_per_timestep = 1
        step_between_insertion = 1

        good_map = None
        good_task_list = None
        good_start_pos_list = None
        good_parking_spot_list = None
        obtained_good = False

        for i in range(repetition):
            # map creation
            grid_map = create_random_grid_map(map_shape=shape, map_density=density, connected=True)

            # non task endpoints
            # non task endpoints, task list
            start_pos_list, non_task_ep_list, task_list = get_start_pos_non_tep_task_list(input_map=grid_map,
                                                                                          agent_num=agent_num,
                                                                                          task_num=task_num)
            parking_spot_list = list(set(non_task_ep_list)-set(start_pos_list))

            # TP args: input_map, start_pos_list, task_list, parking_spot,
            #          imm_task_split=0.5, new_task_per_timestep=1, step_between_insertion=1
            p = Process(target=tp, name="TP", args=(grid_map, start_pos_list, task_list, parking_spot_list,
                                                    imm_task_split,
                                                    new_task_per_timestep,
                                                    step_between_insertion))
            p.start()

            # wait for n seconds
            p.join(10)

            # If process is active
            if p.is_alive():
                # Terminate
                p.terminate()
                p.join()    # clean up
                bad_mapd_inst_count += 1
            # tp has finished
            else:
                # relaunch tp to get agent schedule and timer
                start_time = timeit.default_timer()

                agent_schedule = tp(input_map=grid_map,
                                    start_pos_list=start_pos_list, task_list=task_list,
                                    parking_spot_list=parking_spot_list,
                                    imm_task_split=imm_task_split, new_task_per_insertion=new_task_per_timestep,
                                    step_between_insertion=step_between_insertion)

                # write time
                time_diff = timeit.default_timer() - start_time
                time_list.append(time_diff)

                # collect conflicts
                coll_count, _ = count_collision(agent_schedule=agent_schedule)
                collision_count_list.append(coll_count)

                # collect makespan
                makespan_list.append(len(agent_schedule[0]))

                if not obtained_good:
                    good_map = grid_map.copy()
                    good_task_list = task_list.copy()
                    good_parking_spot_list = parking_spot_list.copy()
                    good_start_pos_list = start_pos_list.copy()
                    obtained_good = True

            print(f'Solved scenario {i+1}/{repetition}')

        # run one finishable instance to print out and test
        agent_schedule = tp(input_map=good_map,
                            start_pos_list=good_start_pos_list, task_list=good_task_list,
                            parking_spot_list=good_parking_spot_list,
                            imm_task_split=imm_task_split, new_task_per_insertion=new_task_per_timestep,
                            step_between_insertion=step_between_insertion)

        self.assertIsInstance(agent_schedule, dict)
        length = len(agent_schedule[0])
        for schedule in agent_schedule.values():
            self.assertEqual(length, len(schedule))

        print(f'Agents starting positions: {good_start_pos_list}')
        print('Task List:')
        pprint(good_task_list)
        print('Resulting Schedule:')
        for schedule in agent_schedule.items():
            print(schedule)
        print(f'Average makespan: {statistics.mean(makespan_list)}')
        print(f'Average TP execution time: {statistics.mean(time_list)}')
        print(f'Average collision detected: {collision_count_list}')
        print(f'Bad instances of MAPD: {bad_mapd_inst_count} out of {repetition}')

        # coll_count, collision_time_list = count_collision(agent_schedule=agent_schedule)
        # pprint(good_map)
        # print(f'Collision detected: {coll_count}')
        # print(f'Collision times: {collision_time_list}')


if __name__ == '__main__':
    unittest.main()
