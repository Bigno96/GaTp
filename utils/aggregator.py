"""
Utility Class for aggregating data in DAgger algorithm
"""

import math
import itertools
import logging
import random

import utils.multi_agent_simulator as ma_sim
import utils.metrics as metric
import experts.token_passing as tp
import utils.expert_utils as exp_utils
import data_loading.dataset as dset

from easydict import EasyDict
from typing import List, Tuple
from collections import deque
from threading import Thread
from multiprocessing import Pool, Manager

STOP_SENTINEL = 'STOP'


class Aggregator:
    """
    Used by DAgger
    """

    def __init__(self,
                 config: EasyDict):
        """
        :param config: configuration Namespace
        """
        self.config = config
        self.logger = logging.getLogger('Aggregator')
        # maximum number of instances to add to the dataset each epoch
        self.max_new_instances: int = config.max_new_instances
        # how many challenging timesteps to select, from most generated to collision downward
        self.selection_p: float = config.selection_percentage
        # mode of working: if not training and use dagger, methods becomes NOP
        self.mode = config.mode
        self.use_dagger = config.use_dagger

        self.cases_list: List[EasyDict] = []   # keep tracks of the high difficulty cases

    def collect_cases(self,
                      simulator: ma_sim.MultiAgentSimulator,
                      valid_basename: str,
                      epoch: int
                      ) -> None:
        if self.mode == 'train' and self.use_dagger:
            self.__collect_cases(simulator=simulator,
                                 valid_basename=valid_basename,
                                 epoch=epoch)

    def __collect_cases(self,
                        simulator: ma_sim.MultiAgentSimulator,
                        valid_basename: str,
                        epoch: int
                        ) -> None:
        """
        Collect cases that result in high number of conflict when solved by our policy
        :param simulator: from utils.multi_agent_simulator.py
        :param valid_basename: basename of the validation file over which the model is being tested
        :param epoch: id of the epoch when Dagger is called, used for naming new entries
        """
        # get timesteps at which collision happens and get number of collision for each timestep
        _, coll_time_list, coll_list = metric.count_collision(agent_schedule=simulator.agent_schedule)

        # if no collisions
        if not coll_time_list:
            return

        # removes all timesteps with zero collisions
        coll_list = [x for x in coll_list
                     if x != 0]
        # zip timesteps and collision number at each timestep
        # el[0]: timesteps, el[1]: number of collision
        coll_dict = list(zip(coll_time_list, coll_list))
        # sort by decreasing order of collisions
        coll_dict = list(sorted(coll_dict,
                                key=lambda el: el[1],
                                reverse=True))

        # extract a % of the timesteps with more collisions
        split_idx = int(len(coll_dict) * self.selection_p)
        coll_dict = coll_dict[:split_idx]

        # get a list of the timesteps that precedes those with high number of collisions
        timestep_list = [el[0]-1 for el in coll_dict]
        # loop over all obtained timesteps to get cases data
        for t in timestep_list:
            case = EasyDict()
            # name
            case.basename = f'dagger_e{epoch}_{valid_basename}_t{t}'

            # agents position
            case.agent_pos = [val[t][:-1]   # leave out timestep
                              for key, val in simulator.agent_schedule.items()]
            #  agents goal
            case.agent_goal = [val[t][:-1]   # leave out timestep
                               for key, val in simulator.goal_schedule.items()]
            # task assigned
            case.assigned_task = [list(val[t][:-1])[0]  # leave out timestep
                                  for key, val in simulator.task_schedule.items()]
            # get activated tasks
            num_activated_task = min(int((self.config.imm_task_split
                                          * self.config.task_number)
                                         # starting number of active task
                                         +
                                         # activated task at each timestep
                                         (self.config.new_task_per_timestep
                                          * math.floor(t / self.config.step_between_insertion))
                                         ),
                                     self.config.task_number)
            activated_task = simulator.task_list.tolist()[:num_activated_task]
            # get completed tasks
            completed_task = [list(simulator.task_schedule[i][j][:-1])[0]
                              for i in range(self.config.agent_number)
                              for j in range(t)]
            completed_task.sort()
            completed_task = [task
                              for task, _ in itertools.groupby(completed_task)]    # remove duplicates
            completed_task = [task
                              for task in completed_task
                              if task  # remove empty tasks
                              # remove task currently assigned, that were wrongly considered up to here
                              and task not in case.assigned_task]
            # get active tasks
            case.active_task = [task
                                for task in activated_task
                                if task not in completed_task
                                and task not in case.assigned_task]
            # tasks yet to activate
            case.new_task_pool = simulator.task_list.tolist()[num_activated_task:]
            # get map
            case.map = simulator.map

            # tuple conversion
            case.assigned_task = task_to_tuple(task_list=case.assigned_task)
            case.active_task = task_to_tuple(task_list=case.active_task)
            case.new_task_pool = deque(task_to_tuple(task_list=case.new_task_pool))

            # append to case list
            self.cases_list.append(case)

    def extend_dataset(self,
                       dataset: dset.GaTpDataset
                       ) -> None:
        if self.mode == 'train' and self.use_dagger:
            self.__extend_dataset(dataset=dataset)

    def __extend_dataset(self,
                         dataset: dset.GaTpDataset
                         ) -> None:
        """
        Extend the training dataset calling Expert on all the collected cases
        :param dataset: instance of the train dataset to extend with new entries
        """
        if not self.max_new_instances:  # return if 0 new instances
            return

        self.logger.info('Generate new Training data')
        manager = Manager()

        # set up queues for returning values
        map_queue = manager.Queue(maxsize=self.max_new_instances+1)
        expert_sol_queue = manager.Queue(maxsize=self.max_new_instances+1)
        basename_queue = manager.Queue(maxsize=self.max_new_instances+1)

        # choose expert type to run
        if self.config.expert_type == 'tp':
            worker = OnlineTpWorker(config=self.config,
                                    map_queue=map_queue,
                                    expert_sol_queue=expert_sol_queue,
                                    basename_queue=basename_queue)
        else:
            raise ValueError('Invalid expert selected')

        # run experts only over maximum number of cases
        self.cases_list = random.sample(population=self.cases_list,
                                        # don't pick more cases than available ones
                                        k=min(self.max_new_instances, len(self.cases_list)))

        # spawn and run processes
        with Pool() as pool:
            pool.map(func=worker, iterable=self.cases_list)

        # convert queues to lists
        map_queue.put(STOP_SENTINEL, block=True)  # termination sentinel
        map_list = [p for p in iter(map_queue.get, STOP_SENTINEL)]
        expert_sol_queue.put(STOP_SENTINEL, block=True)  # termination sentinel
        expert_sol_list = [p for p in iter(expert_sol_queue.get, STOP_SENTINEL)]
        basename_queue.put(STOP_SENTINEL, block=True)  # termination sentinel
        basename_list = [p for p in iter(basename_queue.get, STOP_SENTINEL)]

        # finally, extend dataset
        dataset.extend_dataset(input_map_list=map_list,
                               expert_sol_list=expert_sol_list,
                               basename_list=basename_list)


class OnlineTpWorker:
    """
    Worker for Online Token Passing, called during Dataset Aggregation
    """

    def __init__(self,
                 config: EasyDict,
                 map_queue: Manager,
                 expert_sol_queue: Manager,
                 basename_queue: Manager):
        """
        :param config: Namespace of dataset configurations
        :param map_queue: sharing map names
        :param expert_sol_queue: sharing computed expert solutions
        :param basename_queue: sharing basename for naming new instances
        """
        self.config = config
        self.map_queue = map_queue
        self.expert_sol_queue = expert_sol_queue
        self.basename_queue = basename_queue

    def __call__(self,
                 case: EasyDict
                 ) -> None:
        """
        :param case:
                {'map': created map,
                 'agent_pos: agents starting positions,
                 'new_task_pool': list of tasks waiting for activation,
                 'active_task': list of active tasks,
                 'assigned_task': list of tasks assigned to the agent at the start}
        """
        # setup return structures
        agent_schedule = {}
        goal_schedule = {}
        metrics = {}
        execution = exp_utils.StopToken()

        # run online tp
        kwargs = {'input_map': case.map,
                  'start_pos_list': case.agent_pos,
                  'new_task_pool': case.new_task_pool,
                  'active_task_list': case.active_task,
                  'assigned_task_list': case.assigned_task,
                  'agent_schedule': agent_schedule,
                  'goal_schedule': goal_schedule,
                  'metrics': metrics,
                  'execution': execution,
                  'new_task_per_insertion': self.config.new_task_per_timestep,
                  'step_between_insertion': self.config.step_between_insertion}

        # run online token passing
        worker = Thread(target=tp.online_tp, kwargs=kwargs)
        worker.start()

        # wait for timeout
        worker.join(self.config.timeout)

        # bad MAPD, doesn't terminate
        if worker.is_alive():
            execution.cancel()  # cancel the execution
            worker.join()
        # expert naturally terminates
        else:
            # add results to the sharing queues
            expert_sol = {'agent_schedule': agent_schedule,
                          'goal_schedule': goal_schedule,
                          'metrics': metrics}

            self.map_queue.put(case.map, block=True)
            self.expert_sol_queue.put(expert_sol, block=True)
            self.basename_queue.put(case.basename, block=True)


def task_to_tuple(task_list: List[List[List[int]]]
                  ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    :param task_list: task list to convert
    :return: list with tasks converted as tuples
    """
    ret = []
    for task in task_list:
        if not task:
            ret.append(())
        else:
            pickup, location = task
            ret.append((tuple(pickup), tuple(location)))

    return ret
