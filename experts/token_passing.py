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

from experts.tp_agent import TpAgent
from utils.expert_utils import preprocess_heuristics


def tp(input_map, start_pos_list, task_list, parking_spot_list,
       agent_schedule, metrics, execution,
       imm_task_split=0, new_task_per_insertion=1, step_between_insertion=1):
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
    :param agent_schedule, dict, RETURN VALUE
             agent_schedule -> {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
    :param metrics, dict, RETURN VALUE
             service_time, list of float, number of timesteps for completing each task
             timestep_runtime, list of float, execution time of each timestep, in ms
    :param execution: StopToken instance, used to terminate hanging instances
            when using stop, return values behaviour is undefined
    """
    # starting positions are used as non-task endpoints
    non_task_ep_list = start_pos_list + parking_spot_list

    # precompute heuristics maps towards all endpoints
    h_coll = preprocess_heuristics(input_map=input_map,
                                   task_list=task_list, non_task_ep_list=non_task_ep_list)

    # instantiate agents and free agent queue
    agent_pool = {TpAgent(name=idx, input_map=input_map,
                          start_pos=start_pos_list[idx], h_coll=h_coll)
                  for idx in range(len(start_pos_list))
                  }
    free_agent_queue = deque()

    # instantiate token, dict -> {agent_name : {pos: (x,y), path: [(x,y,t), ...]}}
    token = {}
    for agent in agent_pool:
        token[agent.name] = {'pos': agent.pos,
                             'path': agent.path.copy()}

    # set up a list of active, immediately available tasks and a pool of 'new' tasks
    # split done according to imm_task_split value
    split_idx = int(imm_task_split*len(task_list))
    active_task_list = task_list[:split_idx]
    new_task_pool = deque(task_list[split_idx:])

    # set up checks for how many tasks have been injected
    activated_task_count = len(active_task_list)
    total_task_count = len(task_list)

    # set up agent_schedule
    for agent in agent_pool:
        agent_schedule[agent.name] = agent.path.copy()
        agent.move_one_step()       # move them to timestep 1

    # track time and metrics
    timestep = 1            # timestep = 0 is the initialization
    metrics['service_time'] = []
    metrics['timestep_runtime'] = []

    # while tasks are available or at least one agent is still busy
    while active_task_list or activated_task_count < total_task_count \
            or not all(ag.is_free for ag in agent_pool):

        # exit condition, avoid hanging
        # undefined return behaviour
        if execution.is_cancelled:
            return

        # add new tasks, if any, at the start of the iteration
        if (timestep % step_between_insertion) == 0:  # every n step add new tasks
            new_task_list = [new_task_pool.popleft()
                             for _ in range(min(len(new_task_pool), new_task_per_insertion))
                             ]
            active_task_list.extend(new_task_list)
            activated_task_count += len(new_task_list)

        # timing for metrics
        start_time = timeit.default_timer()

        # list of free agents that will request the token
        free_agent_queue.extend([agent
                                 for agent in agent_pool
                                 if agent.is_free])

        # while agent a_i exists that requests token
        while free_agent_queue:
            agent = free_agent_queue.popleft()
            # pass control to agent a_i
            # a_i updates token, active_task_list and its 'free' status
            sel_task = agent.receive_token(token=token,
                                           task_list=active_task_list,
                                           non_task_ep_list=non_task_ep_list,
                                           sys_timestep=timestep)
            # if a task was assigned
            if sel_task:
                metrics['service_time'].append(len(agent.path))     # add path length to complete the task

        # check for eventual collisions and adapt
        for agent in agent_pool:
            agent.collision_shielding(token=token, sys_timestep=timestep, agent_pool=agent_pool)

        # all agents move along their paths in token for one timestep
        for agent in agent_pool:
            # update schedule
            agent_schedule[agent.name].append(agent.path[0])
            # agents update here if they are free or not (when they end a path, they become free)
            agent.move_one_step()
            # update token
            token[agent.name]['pos'] = agent.pos

        # update timings
        elapsed_time = timeit.default_timer() - start_time
        metrics['timestep_runtime'].append(elapsed_time * 1000)  # ms conversion

        timestep += 1


'''if __name__ == '__main__':
    __spec__ = None

    import numpy as np

    i_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=np.int8)

    agents_starting_positions = [(8, 17), (5, 8), (8, 4), (16, 19), (6, 15), (13, 14), (12, 14), (4, 18), (14, 4),
                                 (11, 1), (8, 9), (9, 10), (12, 4), (9, 6), (10, 14), (12, 5), (0, 13), (2, 5),
                                 (5, 4), (17, 9)]

    t_list = [((5, 13), (17, 15)),
                 ((18, 16), (11, 11)),
                 ((2, 17), (3, 14)),
                 ((13, 8), (6, 2)),
                 ((10, 18), (17, 6)),
                 ((10, 3), (11, 8)),
                 ((14, 10), (12, 8)),
                 ((8, 13), (16, 15)),
                 ((4, 6), (9, 16)),
                 ((3, 6), (18, 4)),
                 ((15, 14), (14, 17)),
                 ((4, 17), (2, 12)),
                 ((2, 1), (13, 13)),
                 ((5, 12), (17, 11)),
                 ((6, 12), (13, 15)),
                 ((11, 16), (9, 11)),
                 ((11, 7), (10, 12)),
                 ((9, 2), (4, 11)),
                 ((7, 18), (2, 17)),
                 ((7, 18), (3, 18)),
                 ((2, 18), (15, 8)),
                 ((5, 16), (2, 14)),
                 ((5, 3), (5, 10)),
                 ((11, 7), (17, 6)),
                 ((6, 1), (11, 12)),
                 ((9, 17), (10, 3)),
                 ((14, 18), (4, 6)),
                 ((12, 16), (8, 13)),
                 ((1, 5), (4, 12)),
                 ((1, 9), (10, 7)),
                 ((1, 4), (16, 18)),
                 ((7, 2), (7, 1)),
                 ((3, 2), (5, 17)),
                 ((10, 11), (7, 11)),
                 ((5, 11), (6, 16)),
                 ((8, 10), (4, 2)),
                 ((15, 18), (6, 5)),
                 ((2, 14), (1, 15)),
                 ((15, 14), (8, 12)),
                 ((15, 6), (13, 15)),
                 ((5, 15), (1, 8)),
                 ((9, 4), (12, 6)),
                 ((16, 11), (17, 3)),
                 ((9, 9), (2, 2)),
                 ((3, 9), (6, 8)),
                 ((15, 4), (9, 13)),
                 ((2, 15), (8, 7)),
                 ((13, 8), (1, 15)),
                 ((14, 13), (13, 3)),
                 ((6, 5), (1, 5)),
                 ((11, 11), (5, 10)),
                 ((17, 3), (18, 7)),
                 ((8, 15), (13, 17)),
                 ((15, 4), (10, 16)),
                 ((9, 12), (2, 16)),
                 ((5, 1), (6, 4)),
                 ((17, 1), (3, 8)),
                 ((2, 8), (12, 16)),
                 ((12, 17), (1, 9)),
                 ((2, 16), (10, 2)),
                 ((3, 6), (18, 11)),
                 ((6, 4), (7, 8)),
                 ((5, 2), (1, 3)),
                 ((15, 12), (1, 2)),
                 ((18, 9), (8, 6)),
                 ((13, 6), (1, 18)),
                 ((8, 11), (1, 2)),
                 ((5, 12), (7, 7)),
                 ((10, 13), (10, 3)),
                 ((11, 18), (13, 17)),
                 ((17, 3), (16, 9)),
                 ((13, 8), (11, 13)),
                 ((15, 14), (5, 16)),
                 ((4, 10), (1, 5)),
                 ((15, 8), (12, 2)),
                 ((3, 14), (11, 5)),
                 ((7, 3), (1, 8)),
                 ((15, 1), (6, 5)),
                 ((17, 5), (10, 18)),
                 ((9, 7), (14, 17)),
                 ((2, 6), (3, 10)),
                 ((15, 14), (1, 7)),
                 ((6, 1), (1, 11)),
                 ((18, 4), (5, 17)),
                 ((7, 13), (17, 11)),
                 ((15, 6), (18, 5)),
                 ((14, 15), (1, 15)),
                 ((9, 17), (13, 1)),
                 ((16, 17), (5, 10)),
                 ((16, 10), (13, 7)),
                 ((8, 6), (6, 12)),
                 ((5, 9), (9, 16)),
                 ((7, 2), (6, 7)),
                 ((16, 17), (2, 4)),
                 ((17, 3), (6, 13)),
                 ((11, 5), (10, 9)),
                 ((6, 12), (9, 14)),
                 ((10, 5), (13, 7)),
                 ((9, 16), (17, 1)),
                 ((12, 17), (13, 13)),
                 ((4, 3), (13, 15)),
                 ((18, 7), (16, 11)),
                 ((5, 15), (13, 13)),
                 ((6, 11), (11, 5)),
                 ((1, 7), (8, 8)),
                 ((13, 11), (16, 17)),
                 ((3, 7), (9, 14)),
                 ((5, 16), (13, 11)),
                 ((4, 2), (9, 7)),
                 ((13, 1), (7, 4)),
                 ((6, 12), (14, 9)),
                 ((4, 10), (3, 11)),
                 ((5, 12), (3, 8)),
                 ((9, 17), (3, 7)),
                 ((1, 14), (11, 9)),
                 ((3, 2), (6, 8)),
                 ((18, 10), (17, 17)),
                 ((1, 4), (15, 8)),
                 ((10, 13), (13, 6)),
                 ((15, 6), (17, 13)),
                 ((7, 6), (5, 10)),
                 ((8, 11), (7, 18)),
                 ((12, 17), (7, 17)),
                 ((7, 16), (2, 17)),
                 ((18, 17), (5, 5)),
                 ((4, 2), (3, 16)),
                 ((11, 5), (12, 2)),
                 ((11, 9), (6, 17)),
                 ((16, 5), (12, 10)),
                 ((16, 5), (12, 11)),
                 ((12, 2), (12, 9)),
                 ((6, 16), (3, 5)),
                 ((8, 6), (5, 10)),
                 ((10, 4), (12, 17)),
                 ((17, 7), (7, 7)),
                 ((10, 15), (9, 7)),
                 ((14, 7), (3, 3)),
                 ((18, 11), (11, 6)),
                 ((13, 13), (9, 4)),
                 ((7, 8), (5, 7)),
                 ((15, 4), (4, 10)),
                 ((8, 12), (15, 15)),
                 ((11, 17), (7, 12)),
                 ((6, 5), (15, 2)),
                 ((16, 13), (2, 17)),
                 ((8, 8), (14, 17)),
                 ((3, 14), (8, 10)),
                 ((18, 17), (7, 14)),
                 ((5, 9), (12, 8)),
                 ((18, 18), (12, 10)),
                 ((15, 16), (11, 11)),
                 ((17, 15), (18, 4)),
                 ((13, 6), (11, 4)),
                 ((14, 17), (3, 7)),
                 ((1, 18), (13, 9)),
                 ((9, 2), (7, 15)),
                 ((18, 13), (4, 13)),
                 ((14, 2), (13, 4)),
                 ((14, 15), (15, 9)),
                 ((4, 3), (1, 9)),
                 ((11, 11), (5, 16)),
                 ((18, 17), (3, 14)),
                 ((15, 6), (18, 9)),
                 ((16, 11), (16, 6)),
                 ((18, 5), (5, 7)),
                 ((9, 17), (5, 6)),
                 ((15, 11), (7, 15)),
                 ((10, 12), (11, 6)),
                 ((9, 4), (8, 15)),
                 ((6, 9), (6, 7)),
                 ((1, 10), (15, 9)),
                 ((8, 18), (16, 1)),
                 ((17, 18), (18, 9)),
                 ((5, 17), (2, 9)),
                 ((18, 4), (1, 14)),
                 ((9, 14), (3, 16)),
                 ((17, 6), (12, 10)),
                 ((17, 13), (5, 3)),
                 ((3, 2), (2, 6)),
                 ((1, 17), (2, 18)),
                 ((16, 11), (8, 7)),
                 ((7, 10), (1, 2)),
                 ((9, 14), (16, 1)),
                 ((9, 5), (4, 12)),
                 ((18, 18), (6, 11)),
                 ((6, 6), (4, 17)),
                 ((13, 4), (6, 8)),
                 ((3, 12), (10, 7)),
                 ((13, 18), (8, 15)),
                 ((9, 11), (18, 12)),
                 ((5, 17), (4, 8)),
                 ((18, 10), (1, 7)),
                 ((15, 4), (11, 3)),
                 ((2, 8), (13, 8)),
                 ((13, 17), (16, 4)),
                 ((5, 5), (17, 12)),
                 ((1, 18), (6, 11)),
                 ((9, 14), (8, 15)),
                 ((16, 18), (1, 5)),
                 ((17, 18), (18, 6)),
                 ((4, 7), (17, 4)),
                 ((10, 5), (10, 10)),
                 ((10, 8), (12, 6)),
                 ((16, 9), (10, 15)),
                 ((13, 4), (1, 4)),
                 ((18, 6), (3, 9)),
                 ((17, 17), (15, 7)),
                 ((6, 8), (2, 7)),
                 ((4, 4), (1, 18)),
                 ((12, 7), (2, 6)),
                 ((16, 17), (3, 8)),
                 ((4, 5), (9, 13)),
                 ((2, 16), (5, 18)),
                 ((9, 17), (1, 16)),
                 ((6, 10), (6, 9)),
                 ((2, 11), (2, 7)),
                 ((3, 11), (11, 5)),
                 ((3, 1), (14, 2)),
                 ((11, 15), (6, 12)),
                 ((12, 10), (8, 12)),
                 ((17, 4), (8, 18)),
                 ((16, 14), (17, 14)),
                 ((6, 2), (15, 12)),
                 ((12, 10), (4, 14)),
                 ((16, 18), (12, 10)),
                 ((4, 15), (4, 6)),
                 ((2, 18), (10, 5)),
                 ((3, 2), (10, 1)),
                 ((15, 10), (12, 13)),
                 ((3, 14), (11, 9)),
                 ((6, 10), (11, 12)),
                 ((3, 8), (13, 15)),
                 ((16, 7), (1, 6)),
                 ((18, 3), (5, 5)),
                 ((17, 2), (6, 17)),
                 ((1, 6), (13, 15)),
                 ((8, 3), (14, 15)),
                 ((9, 2), (7, 2)),
                 ((1, 10), (6, 17)),
                 ((16, 17), (14, 18)),
                 ((17, 17), (16, 1)),
                 ((14, 14), (9, 18)),
                 ((8, 6), (10, 7)),
                 ((5, 5), (17, 6)),
                 ((7, 17), (14, 3)),
                 ((2, 17), (6, 1)),
                 ((13, 7), (5, 5)),
                 ((16, 11), (7, 4)),
                 ((3, 5), (16, 13)),
                 ((6, 13), (17, 2)),
                 ((15, 13), (10, 11)),
                 ((14, 1), (18, 5)),
                 ((11, 15), (7, 15)),
                 ((1, 10), (14, 5)),
                 ((16, 18), (16, 6)),
                 ((3, 1), (15, 6)),
                 ((1, 14), (17, 10)),
                 ((9, 17), (12, 17)),
                 ((9, 7), (15, 8)),
                 ((14, 17), (18, 3)),
                 ((13, 17), (5, 3)),
                 ((5, 17), (6, 3)),
                 ((5, 1), (17, 12)),
                 ((12, 10), (14, 13)),
                 ((2, 7), (15, 18)),
                 ((7, 8), (4, 2)),
                 ((16, 6), (6, 16)),
                 ((6, 11), (3, 1)),
                 ((17, 3), (11, 11)),
                 ((15, 15), (14, 1)),
                 ((16, 6), (17, 15)),
                 ((3, 6), (13, 15)),
                 ((18, 7), (7, 10)),
                 ((7, 4), (3, 14)),
                 ((18, 7), (1, 5)),
                 ((8, 14), (9, 9)),
                 ((11, 18), (13, 4)),
                 ((8, 6), (9, 4)),
                 ((15, 8), (15, 3)),
                 ((10, 10), (11, 3)),
                 ((11, 6), (13, 8)),
                 ((13, 12), (2, 17)),
                 ((5, 13), (9, 12)),
                 ((5, 6), (12, 7)),
                 ((10, 15), (4, 11)),
                 ((10, 10), (4, 2)),
                 ((2, 12), (3, 14)),
                 ((7, 10), (6, 16)),
                 ((2, 18), (6, 17)),
                 ((17, 15), (15, 12)),
                 ((17, 12), (11, 18)),
                 ((13, 11), (13, 8)),
                 ((7, 13), (18, 9)),
                 ((17, 15), (5, 17)),
                 ((7, 13), (12, 8)),
                 ((12, 10), (8, 7)),
                 ((13, 7), (12, 17)),
                 ((12, 17), (3, 17)),
                 ((1, 17), (14, 5)),
                 ((1, 7), (18, 18)),
                 ((4, 8), (14, 1)),
                 ((17, 1), (10, 13)),
                 ((4, 1), (14, 14)),
                 ((9, 13), (11, 8)),
                 ((2, 7), (3, 16)),
                 ((11, 6), (4, 12)),
                 ((3, 6), (7, 3)),
                 ((5, 17), (1, 15)),
                 ((4, 4), (6, 8)),
                 ((6, 2), (8, 6)),
                 ((4, 8), (1, 12)),
                 ((16, 14), (13, 12)),
                 ((15, 16), (1, 9)),
                 ((12, 10), (18, 4)),
                 ((6, 3), (13, 3)),
                 ((2, 4), (14, 18)),
                 ((5, 17), (16, 1)),
                 ((7, 12), (10, 7)),
                 ((5, 10), (11, 7)),
                 ((9, 5), (11, 3)),
                 ((17, 11), (11, 9)),
                 ((5, 5), (7, 3)),
                 ((3, 6), (7, 3)),
                 ((1, 15), (3, 16)),
                 ((18, 13), (10, 8)),
                 ((9, 7), (5, 13)),
                 ((7, 4), (10, 13)),
                 ((17, 17), (10, 16)),
                 ((17, 11), (16, 15)),
                 ((15, 2), (16, 17)),
                 ((2, 6), (4, 3)),
                 ((14, 3), (18, 11)),
                 ((4, 8), (7, 13)),
                 ((16, 14), (2, 3)),
                 ((8, 12), (2, 6)),
                 ((15, 17), (1, 4)),
                 ((8, 18), (6, 12)),
                 ((15, 11), (8, 18)),
                 ((4, 2), (5, 2)),
                 ((15, 6), (16, 6)),
                 ((18, 7), (6, 2)),
                 ((8, 15), (9, 11)),
                 ((11, 16), (16, 13)),
                 ((18, 10), (13, 4)),
                 ((6, 11), (1, 2)),
                 ((13, 12), (10, 13)),
                 ((5, 2), (7, 18)),
                 ((5, 11), (10, 16)),
                 ((6, 13), (10, 1)),
                 ((11, 17), (18, 1)),
                 ((1, 17), (7, 1)),
                 ((1, 11), (9, 13)),
                 ((9, 1), (1, 16)),
                 ((3, 6), (11, 15)),
                 ((7, 16), (7, 15)),
                 ((11, 5), (7, 16)),
                 ((15, 7), (1, 8)),
                 ((4, 2), (6, 12)),
                 ((16, 9), (4, 4)),
                 ((3, 6), (18, 7)),
                 ((8, 5), (6, 12)),
                 ((10, 5), (13, 15)),
                 ((3, 10), (10, 4)),
                 ((6, 14), (13, 10)),
                 ((9, 4), (7, 15)),
                 ((2, 15), (9, 11)),
                 ((13, 7), (16, 18)),
                 ((5, 15), (15, 2)),
                 ((16, 14), (18, 3)),
                 ((6, 12), (7, 14)),
                 ((1, 12), (1, 2)),
                 ((14, 3), (18, 11)),
                 ((4, 4), (1, 18)),
                 ((5, 2), (5, 6)),
                 ((18, 9), (9, 14)),
                 ((10, 8), (13, 13)),
                 ((11, 6), (5, 17)),
                 ((16, 3), (10, 15)),
                 ((16, 4), (2, 3)),
                 ((3, 14), (16, 1)),
                 ((1, 6), (7, 4)),
                 ((13, 13), (10, 1)),
                 ((4, 1), (8, 10)),
                 ((9, 11), (11, 15)),
                 ((13, 17), (10, 15)),
                 ((7, 18), (16, 18)),
                 ((18, 11), (1, 17)),
                 ((2, 17), (5, 13)),
                 ((15, 15), (16, 10)),
                 ((6, 8), (9, 17)),
                 ((16, 12), (16, 9)),
                 ((1, 18), (18, 11)),
                 ((2, 12), (18, 16)),
                 ((4, 9), (16, 3)),
                 ((17, 10), (5, 10)),
                 ((16, 18), (8, 10)),
                 ((15, 9), (11, 4)),
                 ((11, 5), (1, 11)),
                 ((9, 11), (1, 2)),
                 ((1, 5), (13, 18)),
                 ((9, 17), (15, 16)),
                 ((16, 6), (5, 13)),
                 ((8, 12), (7, 3)),
                 ((9, 14), (13, 2)),
                 ((6, 2), (10, 9)),
                 ((7, 11), (13, 8)),
                 ((15, 10), (8, 15)),
                 ((9, 2), (14, 2)),
                 ((9, 18), (13, 10)),
                 ((3, 1), (1, 15)),
                 ((16, 1), (14, 18)),
                 ((12, 7), (17, 15)),
                 ((4, 1), (13, 3)),
                 ((7, 11), (2, 17)),
                 ((18, 5), (14, 5)),
                 ((15, 15), (7, 7)),
                 ((11, 2), (12, 10)),
                 ((8, 10), (15, 8)),
                 ((5, 3), (14, 16)),
                 ((18, 17), (17, 5)),
                 ((17, 4), (3, 13)),
                 ((18, 16), (18, 3)),
                 ((1, 12), (4, 10)),
                 ((15, 10), (6, 1)),
                 ((6, 8), (13, 6)),
                 ((1, 12), (8, 5)),
                 ((2, 6), (13, 3)),
                 ((3, 7), (16, 18)),
                 ((5, 17), (16, 18)),
                 ((2, 7), (10, 18)),
                 ((18, 4), (4, 13)),
                 ((2, 14), (15, 15)),
                 ((17, 4), (5, 16)),
                 ((1, 3), (3, 8)),
                 ((1, 2), (10, 10)),
                 ((7, 10), (4, 13)),
                 ((5, 6), (18, 6)),
                 ((6, 4), (13, 7)),
                 ((4, 4), (9, 11)),
                 ((16, 14), (14, 7)),
                 ((12, 11), (4, 4)),
                 ((1, 15), (17, 18)),
                 ((9, 1), (11, 6)),
                 ((4, 10), (8, 7)),
                 ((16, 18), (3, 16)),
                 ((18, 4), (1, 17)),
                 ((16, 6), (18, 1)),
                 ((1, 5), (7, 14)),
                 ((11, 12), (10, 10)),
                 ((9, 17), (3, 11)),
                 ((11, 7), (17, 2)),
                 ((11, 14), (10, 18)),
                 ((13, 9), (16, 9)),
                 ((1, 17), (3, 13)),
                 ((3, 6), (17, 17)),
                 ((6, 18), (16, 12)),
                 ((15, 11), (3, 1)),
                 ((14, 17), (16, 9)),
                 ((9, 9), (4, 6)),
                 ((3, 1), (12, 6)),
                 ((15, 7), (14, 1)),
                 ((12, 16), (4, 12)),
                 ((17, 2), (17, 1)),
                 ((16, 5), (13, 11)),
                 ((2, 13), (6, 12)),
                 ((7, 18), (3, 10)),
                 ((13, 8), (14, 3)),
                 ((5, 2), (11, 18)),
                 ((18, 12), (10, 12)),
                 ((10, 4), (17, 2)),
                 ((6, 8), (9, 2)),
                 ((17, 2), (15, 10)),
                 ((16, 15), (2, 18)),
                 ((12, 2), (2, 2)),
                 ((16, 11), (3, 7)),
                 ((11, 9), (11, 7)),
                 ((8, 8), (12, 10)),
                 ((7, 12), (15, 12)),
                 ((18, 11), (9, 7)),
                 ((4, 14), (6, 11)),
                 ((3, 5), (14, 3)),
                 ((7, 12), (3, 4)),
                 ((5, 12), (13, 15)),
                 ((6, 4), (12, 9)),
                 ((17, 3), (18, 9)),
                 ((15, 18), (3, 2)),
                 ((14, 9), (16, 5)),
                 ((15, 8), (16, 5)),
                 ((16, 4), (4, 7)),
                 ((13, 10), (6, 18)),
                 ((14, 15), (5, 18)),
                 ((9, 18), (3, 12)),
                 ((2, 8), (13, 18)),
                 ((17, 14), (18, 1)),
                 ((17, 2), (5, 12)),
                 ((16, 7), (15, 10)),
                 ((18, 8), (9, 16)),
                 ((1, 7), (10, 3)),
                 ((5, 15), (18, 5)),
                 ((3, 3), (9, 7))]

    from utils.expert_utils import StopToken

    ex = StopToken()
    agent_sched = {}
    met = {}

    tp(input_map=i_map, start_pos_list=agents_starting_positions, task_list=t_list,
       parking_spot_list=[], agent_schedule=agent_sched, metrics=met, execution=ex)

    from utils.metrics import count_collision
    print(count_collision(agent_schedule=agent_sched))
    for sched in agent_sched.items():
        print(sched)'''
