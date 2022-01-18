"""
Token Passing algorithm

Number of agents cooperating is defined as a function parameter
Path1, Path2 functions in the pseudocode are, respectively:
    - experts.a_star
    - TPAgent.find_resting_pos

Agents, once finished their task, if not assigned a new one, moves to a non-conflicting position
That is, an endpoint such that the delivery locations of all tasks are different from the chosen endpoint,
no path of other agents in the token ends in the chosen endpoint
and does not collide with the paths of other agents stored in the token

The following implementation is based on:
    - Token Passing pseudocode as described in
        Ma, H., Li, J., Kumar, T. K., & Koenig, S. (2017).
        Lifelong multiagent path finding for online pickup and delivery tasks.
        arXiv preprint arXiv:1705.10868.
"""
from collections import deque
import timeit
import statistics

from experts.tp_agent import TpAgent
from utils.expert_utils import preprocess_heuristics


def tp(input_map, start_pos_list, task_list, parking_spot_list,
       imm_task_split=0.5, new_task_per_insertion=1, step_between_insertion=1):
    """
    Token Passing algorithm
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param start_pos_list: list of tuples, (x,y) -> coordinates over the map
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param parking_spot_list: list of tuples, (x,y) -> coordinates over the map
           Non-task endpoints for the agent to rest on to avoid deadlocks, can be empty
    :param imm_task_split: float, 1 > x > 0, % of task_list to add to active_task
    :param new_task_per_insertion: int, > 0, how many 'new' task from task_list to add to active_task at each insertion
    :param step_between_insertion: int, > 0, how many timestep between each insertion
    :return: agent_schedule
             agent_schedule -> {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
             service_time, float, average number of timesteps for completing a task
             timestep_runtime, float, average execution time of a timestep, in ms
    """
    # starting positions are used as non-task endpoints
    non_task_ep_list = start_pos_list + parking_spot_list

    # precompute heuristics maps towards all endpoints
    h_coll = preprocess_heuristics(input_map=input_map,
                                   task_list=task_list, non_task_ep_list=non_task_ep_list)

    # instantiate agents
    agent_pool = {TpAgent(name=idx, input_map=input_map,
                          start_pos=start_pos_list[idx], h_coll=h_coll)
                  for idx in range(len(start_pos_list))
                  }
    agent_name_pool = {agent.name for agent in agent_pool}

    # instantiate token, dict -> {agent_id : path}
    token = dict(zip(agent_name_pool, [agent.path for agent in agent_pool]))

    # set up a list of active, immediately available tasks and a pool of 'new' tasks
    # split done according to imm_task_split value
    split_idx = int(imm_task_split*len(task_list))
    active_task_list = task_list[:split_idx]
    new_task_pool = deque(task_list[split_idx:])

    # set up checks for how many tasks have been injected
    activated_task_count = len(active_task_list)
    total_task_count = len(task_list)

    # set up agent_schedule
    agent_schedule = {}
    for name in agent_name_pool:
        agent_schedule[name] = []

    # track time and metrics
    timestep = 0
    metrics = {'service_time': [],
               'timestep_runtime': []}

    # while tasks are available or at least one agent is still busy
    while active_task_list or activated_task_count < total_task_count \
            or not all(ag.is_free for ag in agent_pool):

        start_time = timeit.default_timer()     # timing for metrics

        # list of free agents that will request the token
        free_agent_queue = deque([agent
                                  for agent in agent_pool
                                  if agent.is_free])

        # while agent a_i exists that requests token
        while free_agent_queue:
            agent = free_agent_queue.pop()
            # pass control to agent a_i
            # a_i updates token, active_task_list and its 'free' status
            sel_task = agent.receive_token(token=token,
                                           task_list=active_task_list,
                                           non_task_ep_list=non_task_ep_list,
                                           sys_timestep=timestep)
            # if a task was assigned
            if sel_task:
                metrics['service_time'].append(len(agent.path)-1)     # add path length to complete the task

        # check for eventual collisions and adapt
        for agent in agent_pool:
            agent.collision_shielding(token=token, sys_timestep=timestep, agent_pool=agent_pool)

        # all agents move along their paths in token for one timestep
        for agent in agent_pool:
            # update schedule
            agent_schedule[agent.name].append(agent.path[0])
            # agents update also here if they are free or not (when they end a path, they become free)
            agent.move_one_step()

        # add new tasks, if any, before next iteration
        if (timestep % step_between_insertion) == 0:    # every n step add new tasks
            new_task_list = [new_task_pool.popleft()
                             for _ in range(min(len(new_task_pool), new_task_per_insertion))
                             ]
            active_task_list.extend(new_task_list)
            activated_task_count += len(new_task_list)
        timestep += 1

        elapsed_time = timeit.default_timer() - start_time
        metrics['timestep_runtime'].append(elapsed_time)

    return agent_schedule, \
           statistics.mean(metrics['service_time']), \
           statistics.mean(metrics['timestep_runtime'])*1000


import numpy as np
i_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
       [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
       [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
       [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]])

Agents_starting_positions = [(14, 4), (7, 17), (0, 9), (2, 4), (8, 14), (14, 0), (1, 4), (7, 10), (1, 19), (13, 12), (13, 3), (9, 13), (17, 8), (17, 10), (17, 17), (2, 12), (14, 3), (17, 16), (2, 5), (13, 11)]
Task_List = [((18, 9), (4, 2)),
 ((12, 8), (2, 10)),
 ((16, 4), (0, 14)),
 ((0, 15), (0, 2)),
 ((19, 15), (17, 13)),
 ((6, 19), (1, 9)),
 ((14, 7), (10, 10)),
 ((15, 1), (6, 1)),
 ((10, 13), (15, 0)),
 ((17, 15), (16, 17)),
 ((7, 15), (11, 4)),
 ((5, 18), (17, 7)),
 ((6, 17), (14, 16)),
 ((0, 17), (8, 16)),
 ((11, 0), (2, 19)),
 ((17, 12), (12, 0)),
 ((5, 5), (16, 4)),
 ((6, 3), (10, 3)),
 ((4, 5), (7, 2)),
 ((8, 4), (5, 14)),
 ((3, 11), (13, 8)),
 ((5, 14), (11, 13)),
 ((12, 3), (10, 17)),
 ((7, 0), (6, 5)),
 ((15, 2), (15, 5)),
 ((7, 7), (8, 4)),
 ((14, 9), (16, 2)),
 ((6, 14), (0, 7)),
 ((6, 2), (8, 4)),
 ((1, 0), (17, 15)),
 ((2, 19), (13, 1)),
 ((14, 2), (16, 0)),
 ((19, 0), (7, 5)),
 ((15, 11), (8, 0)),
 ((13, 13), (7, 15)),
 ((6, 12), (12, 7)),
 ((7, 15), (16, 6)),
 ((11, 7), (1, 9)),
 ((6, 5), (9, 17)),
 ((19, 15), (10, 0)),
 ((10, 1), (19, 12)),
 ((9, 4), (15, 18)),
 ((16, 15), (2, 17)),
 ((19, 18), (17, 1)),
 ((19, 16), (18, 16)),
 ((12, 17), (9, 12)),
 ((5, 1), (15, 6)),
 ((15, 1), (6, 13)),
 ((3, 14), (13, 0)),
 ((7, 16), (12, 1))]

ag_schedule, _, _ = tp(input_map=i_map, start_pos_list=Agents_starting_positions,
                       task_list=Task_List, parking_spot_list=[],
                       imm_task_split=0)

for a in ag_schedule.items():
    print(a)
print('\n\n')

from utils.metrics import count_collision
coll_count, coll_list = count_collision(agent_schedule=ag_schedule)

print(coll_count)
