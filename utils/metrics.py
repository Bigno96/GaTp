"""
FIle for computing metrics and evaluating quality of solutions
"""

import sys
import logging

import utils.multi_agent_simulator as ma_sim
import numpy as np

from statistics import mean
from dataclasses import dataclass
from typing import List, Tuple, Dict, Deque


@dataclass(order=True)
class Performance:
    """
    Data class for evaluating testing performances
    3 criteria: % of completed tasks, collision number and makespan
    """
    completed_task: float = 0.  # % of tasks completed
    # 0 - number of agents collisions
    # negative for comparisons: smaller neg value > higher neg value -> small collision count > high coll count
    collisions_difference: float = -sys.maxsize
    # expert makespan - model makespan
    # higher the value, shorter the model solution is -> better
    makespan_difference: float = sys.maxsize

    def copy(self):
        return Performance(self.completed_task,
                           self.collisions_difference,
                           self.makespan_difference)

    def __repr__(self):
        return f'Performance: completed task = {self.completed_task * 100:.2f}%, ' \
               f'collisions = {-self.collisions_difference:.2f}, ' \
               f'makespan degradation = {-self.makespan_difference * 100:.2f}%'


class PerformanceRecorder:
    """
    Class for computing performances of ML model during testing/validation simulations
    """

    def __init__(self,
                 simulator: ma_sim.MultiAgentSimulator):
        """
        :param simulator: from utils.multi_agent_simulator.py
        """
        self.simulator = simulator

    def evaluate_performance(self,
                             target_makespan: int
                             ) -> Performance:
        """
        Compute metrics using information from current state of the simulator
        :param target_makespan: expert's solution makespan
        :return: computed performance
        """
        # obtain percentage of tasks completed
        agent_active_tasks = sum([1
                                  for v in self.simulator.task_register.values()
                                  if v.size > 0])     # number of tasks assigned to agents
        task_percentage = 1 - ((len(self.simulator.active_task_list) + agent_active_tasks)
                               / self.simulator.task_number)
        # count collisions
        collisions, _ = count_collision(agent_schedule=self.simulator.agent_schedule)
        # get makespan difference %
        # agent makespan = 0 -> difference = 1
        # agent makespan = max length -> difference = -1
        # 1 > -1 -> holds consistency for comparison (makespan = 0 > makespan = max length
        makespan_diff = (target_makespan - len(self.simulator.agent_schedule[0])) / target_makespan

        return Performance(completed_task=task_percentage,
                           collisions_difference=-collisions,  # negative, check explanation in Performance
                           makespan_difference=makespan_diff)


def get_avg_performance(performance_list: List[Performance]
                        ) -> Performance:
    """
    Internal method to compute average performance over a list of performances
    """
    # collect all the metrics
    m = np.array([(p.completed_task, p.collisions_difference, p.makespan_difference)
                  for p in performance_list])
    compl_task = m[:, 0]
    coll = m[:, 1]
    mks = m[:, 2]

    # return a Performance instance
    return Performance(completed_task=mean(compl_task),
                       collisions_difference=mean(coll),
                       makespan_difference=mean(mks))


def print_performance(performance: Performance,
                      mode: str,
                      logger: logging.Logger,
                      case_idx: int,
                      max_size: int
                      ) -> None:
    """
    Internal method to print information about performance
    :param performance: Performance instance to print
    :param mode: 'test' or 'train'
    :param logger: logger used to print the info on
    :param case_idx: index of case referred to the performance
    :param max_size: length of the dataset
    """
    # if testing, update at each simulation
    if mode == 'test':
        logger.info(f'Case {case_idx + 1}: [{case_idx + 1}/{max_size}'
                    f'({100 * (case_idx + 1) / max_size:.0f}%)]\t'
                    f'{performance}')
    # else, validation, update every 5 sim
    else:
        if case_idx % 5 == 0:
            logger.info(f'Case {case_idx}: [{case_idx}/{max_size}'
                        f'({100 * case_idx / max_size:.0f}%)]\t'
                        f'{performance}')


def count_collision(agent_schedule: Dict[int, Deque[Tuple[int, int, int]]]
                    ) -> Tuple[int, List[int]]:
    """
    Get all agent's path for a MAPD instance solution and count collisions
    Collision is caused by either node or swap constraint violations
    :param agent_schedule: {agent_id : schedule}
                            with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
    :return: (number of collision detected,
              list of timesteps when collision happens)
    """
    coll_count = 0

    # first, check node conflicts
    # get a new view of agent schedule where all agents' steps are paired by timestep
    # time_view = [ (s1_0, s2_0, s3_0), (s1_1, s2_1, s3_1), ... ]
    # list of tuples of tuples
    time_view = list(map(lambda v: __drop_ts(v),  # remove timesteps
                         zip(*agent_schedule.values())))

    # get number of repeated steps at each timestep -> node conflict
    coll_list = list(map(lambda v: len(v) - len(set(v)),
                         time_view))
    coll_count += sum(coll_list)
    collision_time_list = [idx for idx, val in enumerate(coll_list) if val != 0]

    # second, check swap conflicts
    if len(next(iter(agent_schedule.values()))) > 1:  # if path has at least one step
        # count how many agents
        agent_num = len(agent_schedule)
        # loop over each pair of time slice with its successor
        for t, el in enumerate(zip(time_view, time_view[1:])):
            # sort each couple of subsequent step, for all agents
            # group them in a list
            # swap_view = [(p_a1_t0, p_a1_t1), (p_a2_t0, p_a2_t1), ...]
            # sorted swap_view: sort each tuple in swap view
            swap_view = list(map(lambda v: tuple(sorted(v)),
                                 zip(el[0], el[1])))
            # since it's sorted, 'set' will remove duplicates -> swap conflict
            diff = agent_num - len(set(swap_view))
            if diff:
                coll_count += diff
                collision_time_list.append(t)

    return coll_count, collision_time_list


def __drop_ts(input_list: List[Tuple[int, int, int]]
              ) -> List[Tuple[int, int]]:
    """
    Drop timesteps from passed list of steps
    :param input_list: list of steps, (x, y, t)
    :return: list of only positions, (x, y)
    """
    ret_l = []
    for el in input_list:
        ret_l.append(el[:-1])

    return ret_l
