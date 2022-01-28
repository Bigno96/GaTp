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
    for name in agent_name_pool:
        agent_schedule[name] = []

    # track time and metrics
    timestep = 0
    metrics['service_time'] = []
    metrics['timestep_runtime'] = []

    # while tasks are available or at least one agent is still busy
    while active_task_list or activated_task_count < total_task_count \
            or not all(ag.is_free for ag in agent_pool):

        # exit condition, avoid hanging
        # undefined return behaviour
        if execution.is_cancelled:
            return

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
        metrics['timestep_runtime'].append(elapsed_time*1000)       # ms conversion


'''if __name__ == '__main__':
    __spec__ = None

    import numpy as np

    i_map = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
          dtype=np.int8)

    agents_starting_positions = [(5, 16), (3, 12), (17, 5), (7, 9), (15, 7), (9, 6),
                                 (10, 19), (17, 0), (9, 3), (9, 11), (18, 8),
                                 (9, 17), (18, 16), (19, 5), (6, 12), (17, 8), (10, 6), (2, 19), (8, 0), (11, 2)]

    t_list = [((1, 15), (5, 17)),
                     ((0, 2), (7, 18)),
                     ((9, 9), (2, 16)),
                     ((2, 4), (7, 11)),
                     ((11, 11), (14, 15)),
                     ((16, 4), (14, 17)),
                     ((14, 14), (14, 9)),
                     ((9, 12), (0, 0)),
                     ((13, 17), (19, 18)),
                     ((18, 7), (1, 15)),
                     ((3, 7), (8, 13)),
                     ((3, 9), (12, 12)),
                     ((2, 15), (13, 7)),
                     ((0, 10), (19, 11)),
                     ((5, 5), (4, 10)),
                     ((8, 13), (17, 10)),
                     ((13, 3), (17, 6)),
                     ((14, 4), (16, 12)),
                     ((4, 8), (10, 14)),
                     ((4, 8), (2, 14)),
                     ((1, 3), (8, 18)),
                     ((14, 3), (2, 14)),
                     ((0, 18), (11, 0)),
                     ((6, 5), (15, 11)),
                     ((3, 2), (8, 9)),
                     ((1, 9), (18, 5)),
                     ((12, 10), (19, 8)),
                     ((19, 8), (13, 6)),
                     ((10, 7), (9, 4)),
                     ((12, 5), (6, 6)),
                     ((1, 9), (15, 19)),
                     ((0, 12), (19, 19)),
                     ((7, 13), (14, 5)),
                     ((19, 11), (12, 3)),
                     ((17, 15), (6, 9)),
                     ((6, 17), (5, 5)),
                     ((5, 7), (13, 6)),
                     ((18, 3), (5, 5)),
                     ((2, 17), (12, 14)),
                     ((10, 17), (1, 4)),
                     ((1, 15), (1, 12)),
                     ((12, 4), (2, 13)),
                     ((12, 0), (8, 13)),
                     ((5, 2), (17, 7)),
                     ((6, 4), (7, 11)),
                     ((2, 4), (19, 0)),
                     ((15, 5), (3, 17)),
                     ((0, 16), (13, 16)),
                     ((16, 8), (10, 3)),
                     ((11, 16), (12, 13)),
                     ((4, 18), (5, 2)),
                     ((0, 6), (19, 8)),
                     ((12, 14), (2, 8)),
                     ((9, 15), (0, 1)),
                     ((0, 15), (10, 1)),
                     ((17, 3), (16, 12)),
                     ((17, 9), (12, 4)),
                     ((3, 2), (7, 10)),
                     ((16, 17), (18, 2)),
                     ((0, 4), (11, 16)),
                     ((0, 8), (11, 4)),
                     ((5, 0), (15, 6)),
                     ((5, 1), (2, 5)),
                     ((9, 2), (18, 13)),
                     ((10, 5), (9, 7)),
                     ((1, 1), (16, 5)),
                     ((5, 18), (1, 3)),
                     ((19, 8), (9, 12)),
                     ((10, 11), (8, 10)),
                     ((4, 18), (0, 12)),
                     ((5, 0), (19, 0)),
                     ((14, 12), (12, 13)),
                     ((16, 9), (10, 2)),
                     ((3, 17), (4, 6)),
                     ((16, 17), (19, 19)),
                     ((17, 3), (0, 6)),
                     ((9, 16), (9, 1)),
                     ((12, 3), (3, 8)),
                     ((7, 4), (14, 14)),
                     ((7, 18), (1, 1)),
                     ((0, 9), (6, 8)),
                     ((3, 5), (6, 7)),
                     ((8, 11), (2, 17)),
                     ((16, 10), (1, 3)),
                     ((9, 5), (7, 5)),
                     ((12, 14), (10, 17)),
                     ((18, 11), (8, 14)),
                     ((5, 1), (4, 17)),
                     ((16, 16), (15, 19)),
                     ((8, 9), (4, 13)),
                     ((16, 9), (18, 6)),
                     ((6, 4), (18, 2)),
                     ((19, 15), (19, 6)),
                     ((15, 4), (15, 17)),
                     ((8, 4), (10, 7)),
                     ((12, 9), (12, 0)),
                     ((14, 14), (12, 17)),
                     ((5, 7), (17, 19)),
                     ((8, 9), (10, 2)),
                     ((14, 14), (12, 8)),
                     ((4, 5), (18, 3)),
                     ((19, 11), (18, 2)),
                     ((13, 4), (16, 10)),
                     ((6, 0), (7, 16)),
                     ((15, 5), (5, 17)),
                     ((13, 4), (4, 12)),
                     ((1, 13), (18, 4)),
                     ((3, 13), (5, 2)),
                     ((10, 10), (8, 1)),
                     ((9, 8), (12, 11)),
                     ((13, 5), (3, 6)),
                     ((2, 8), (19, 2)),
                     ((10, 4), (7, 10)),
                     ((6, 5), (16, 1)),
                     ((15, 15), (9, 8)),
                     ((5, 5), (12, 16)),
                     ((19, 19), (18, 3)),
                     ((2, 11), (3, 0)),
                     ((19, 12), (8, 19)),
                     ((14, 15), (16, 2)),
                     ((6, 2), (7, 4)),
                     ((13, 16), (13, 12)),
                     ((19, 18), (0, 12)),
                     ((9, 7), (11, 5)),
                     ((1, 7), (2, 16)),
                     ((1, 7), (15, 15)),
                     ((1, 15), (19, 9)),
                     ((6, 4), (19, 10)),
                     ((8, 1), (9, 16)),
                     ((9, 7), (5, 12)),
                     ((11, 7), (18, 6)),
                     ((17, 14), (17, 1)),
                     ((4, 2), (4, 4)),
                     ((15, 3), (16, 4)),
                     ((7, 11), (12, 12)),
                     ((14, 13), (5, 18)),
                     ((11, 7), (11, 11)),
                     ((19, 10), (18, 10)),
                     ((17, 15), (12, 17)),
                     ((17, 16), (16, 8)),
                     ((10, 1), (0, 18)),
                     ((10, 16), (10, 1)),
                     ((12, 13), (10, 10)),
                     ((13, 16), (18, 14)),
                     ((6, 2), (1, 5)),
                     ((18, 5), (4, 17)),
                     ((2, 17), (10, 5)),
                     ((14, 15), (6, 16)),
                     ((8, 10), (6, 2)),
                     ((15, 2), (14, 5)),
                     ((2, 8), (4, 15)),
                     ((0, 12), (7, 18)),
                     ((16, 8), (0, 1)),
                     ((16, 6), (5, 13)),
                     ((7, 12), (2, 10)),
                     ((14, 15), (14, 9)),
                     ((9, 2), (2, 7)),
                     ((3, 16), (19, 1)),
                     ((10, 2), (0, 11)),
                     ((10, 0), (18, 7)),
                     ((17, 15), (3, 3)),
                     ((7, 4), (15, 1)),
                     ((12, 17), (12, 12)),
                     ((13, 11), (3, 11)),
                     ((2, 10), (11, 5)),
                     ((15, 18), (11, 5)),
                     ((9, 18), (16, 0)),
                     ((0, 16), (8, 13)),
                     ((13, 19), (8, 13)),
                     ((13, 12), (10, 17)),
                     ((10, 0), (19, 12)),
                     ((0, 1), (0, 0)),
                     ((8, 11), (13, 16)),
                     ((16, 5), (1, 8)),
                     ((15, 14), (9, 19)),
                     ((19, 17), (16, 12)),
                     ((2, 17), (11, 18)),
                     ((15, 15), (0, 2)),
                     ((6, 16), (1, 8)),
                     ((8, 6), (2, 11)),
                     ((12, 6), (17, 10)),
                     ((4, 2), (19, 16)),
                     ((15, 15), (5, 7)),
                     ((13, 12), (16, 13)),
                     ((3, 8), (5, 6)),
                     ((13, 5), (0, 1)),
                     ((17, 15), (1, 1)),
                     ((13, 17), (10, 11)),
                     ((14, 13), (1, 11)),
                     ((1, 7), (9, 15)),
                     ((3, 16), (4, 0)),
                     ((6, 10), (8, 18)),
                     ((1, 12), (1, 17)),
                     ((2, 6), (6, 8)),
                     ((13, 16), (13, 18)),
                     ((16, 11), (14, 6)),
                     ((1, 4), (0, 18)),
                     ((18, 5), (0, 1)),
                     ((12, 18), (9, 7)),
                     ((11, 18), (9, 10)),
                     ((11, 3), (1, 7)),
                     ((8, 19), (3, 2)),
                     ((3, 8), (13, 2)),
                     ((3, 7), (13, 1)),
                     ((16, 5), (3, 16)),
                     ((19, 18), (4, 2)),
                     ((17, 4), (10, 17)),
                     ((2, 18), (5, 3)),
                     ((8, 17), (0, 11)),
                     ((17, 14), (13, 11)),
                     ((17, 6), (0, 3)),
                     ((9, 4), (6, 19)),
                     ((16, 16), (8, 3)),
                     ((9, 7), (8, 7)),
                     ((5, 13), (2, 15)),
                     ((10, 5), (2, 12)),
                     ((16, 5), (12, 12)),
                     ((14, 12), (12, 9)),
                     ((9, 5), (5, 5)),
                     ((17, 1), (17, 17)),
                     ((10, 5), (5, 8)),
                     ((15, 10), (10, 7)),
                     ((14, 13), (18, 15)),
                     ((4, 16), (2, 11)),
                     ((18, 11), (13, 13)),
                     ((7, 0), (12, 12)),
                     ((16, 18), (16, 10)),
                     ((1, 0), (7, 16)),
                     ((2, 15), (14, 15)),
                     ((6, 10), (12, 15)),
                     ((10, 11), (13, 17)),
                     ((10, 5), (14, 3)),
                     ((16, 18), (14, 12)),
                     ((10, 4), (10, 14)),
                     ((13, 6), (13, 12)),
                     ((2, 5), (11, 0)),
                     ((1, 10), (2, 3)),
                     ((16, 3), (1, 5)),
                     ((5, 5), (1, 12)),
                     ((5, 1), (10, 4)),
                     ((2, 8), (19, 6)),
                     ((14, 0), (9, 1)),
                     ((2, 11), (5, 5)),
                     ((6, 2), (17, 10)),
                     ((4, 9), (1, 11)),
                     ((13, 19), (14, 3)),
                     ((1, 1), (11, 11)),
                     ((15, 3), (19, 3)),
                     ((12, 16), (11, 19)),
                     ((18, 6), (9, 12)),
                     ((11, 13), (16, 1)),
                     ((3, 16), (12, 6)),
                     ((11, 8), (3, 13)),
                     ((13, 4), (5, 18)),
                     ((7, 17), (0, 3)),
                     ((7, 1), (11, 6)),
                     ((0, 0), (16, 3)),
                     ((6, 19), (16, 13)),
                     ((13, 6), (14, 0)),
                     ((19, 6), (1, 3)),
                     ((9, 18), (7, 15)),
                     ((16, 6), (9, 0)),
                     ((18, 6), (15, 19)),
                     ((13, 2), (10, 7)),
                     ((10, 12), (15, 15)),
                     ((3, 11), (11, 16)),
                     ((4, 3), (19, 9)),
                     ((5, 17), (2, 18)),
                     ((2, 10), (19, 12)),
                     ((7, 6), (19, 1)),
                     ((17, 3), (13, 8)),
                     ((5, 9), (2, 3)),
                     ((4, 14), (19, 10)),
                     ((18, 1), (7, 0)),
                     ((0, 3), (13, 14)),
                     ((8, 18), (5, 2)),
                     ((6, 5), (8, 14)),
                     ((17, 7), (5, 8)),
                     ((0, 15), (15, 17)),
                     ((4, 2), (7, 10)),
                     ((4, 13), (12, 12)),
                     ((13, 10), (0, 5)),
                     ((4, 17), (3, 11)),
                     ((19, 17), (10, 14)),
                     ((17, 16), (16, 18)),
                     ((12, 12), (4, 8)),
                     ((3, 0), (15, 17)),
                     ((11, 11), (4, 9)),
                     ((15, 15), (7, 6)),
                     ((0, 2), (16, 10)),
                     ((8, 1), (15, 11)),
                     ((14, 6), (19, 6)),
                     ((8, 15), (12, 15)),
                     ((5, 14), (3, 10)),
                     ((11, 19), (14, 4)),
                     ((14, 1), (9, 7)),
                     ((4, 4), (18, 4)),
                     ((14, 1), (4, 9)),
                     ((0, 19), (6, 1)),
                     ((7, 6), (4, 7)),
                     ((4, 11), (10, 15)),
                     ((14, 17), (7, 2)),
                     ((12, 11), (3, 3)),
                     ((16, 17), (14, 6)),
                     ((14, 14), (16, 17)),
                     ((0, 2), (19, 1)),
                     ((18, 6), (12, 13)),
                     ((3, 13), (15, 19)),
                     ((3, 16), (5, 6)),
                     ((8, 1), (18, 15)),
                     ((17, 19), (6, 5)),
                     ((13, 3), (5, 8)),
                     ((11, 1), (6, 8)),
                     ((13, 4), (17, 18)),
                     ((14, 5), (12, 18)),
                     ((1, 3), (10, 3)),
                     ((5, 14), (11, 14)),
                     ((3, 16), (2, 3)),
                     ((15, 3), (4, 17)),
                     ((4, 3), (13, 13)),
                     ((12, 9), (3, 10)),
                     ((5, 4), (0, 16)),
                     ((3, 4), (9, 9)),
                     ((12, 17), (17, 14)),
                     ((3, 10), (3, 7)),
                     ((1, 4), (5, 9)),
                     ((17, 18), (11, 8)),
                     ((5, 13), (19, 9)),
                     ((14, 6), (1, 14)),
                     ((1, 5), (19, 16)),
                     ((5, 15), (6, 6)),
                     ((1, 7), (0, 13)),
                     ((5, 9), (18, 1)),
                     ((12, 18), (5, 17)),
                     ((7, 16), (5, 5)),
                     ((6, 7), (15, 10)),
                     ((6, 2), (17, 6)),
                     ((0, 5), (0, 15)),
                     ((13, 11), (19, 6)),
                     ((7, 15), (17, 2)),
                     ((3, 3), (4, 5)),
                     ((3, 13), (14, 12)),
                     ((4, 1), (8, 15)),
                     ((13, 11), (19, 19)),
                     ((9, 5), (10, 11)),
                     ((1, 7), (4, 18)),
                     ((16, 0), (12, 12)),
                     ((15, 13), (12, 1)),
                     ((1, 10), (8, 18)),
                     ((5, 10), (6, 15)),
                     ((11, 7), (3, 4)),
                     ((6, 16), (16, 19)),
                     ((12, 8), (7, 4)),
                     ((12, 17), (12, 15)),
                     ((1, 6), (4, 2)),
                     ((5, 18), (6, 8)),
                     ((4, 16), (6, 6)),
                     ((8, 7), (1, 8)),
                     ((1, 10), (9, 0)),
                     ((7, 13), (18, 17)),
                     ((1, 12), (19, 9)),
                     ((3, 7), (17, 18)),
                     ((0, 6), (1, 10)),
                     ((7, 2), (7, 13)),
                     ((3, 19), (16, 4)),
                     ((0, 12), (5, 8)),
                     ((3, 6), (8, 7)),
                     ((17, 16), (19, 19)),
                     ((6, 5), (14, 4)),
                     ((11, 10), (9, 0)),
                     ((12, 1), (5, 2)),
                     ((7, 3), (0, 13)),
                     ((17, 15), (15, 13)),
                     ((2, 0), (18, 2)),
                     ((13, 7), (3, 14)),
                     ((15, 4), (3, 11)),
                     ((10, 17), (0, 10)),
                     ((1, 3), (16, 17)),
                     ((1, 0), (2, 10)),
                     ((12, 12), (9, 15)),
                     ((3, 2), (19, 8)),
                     ((4, 17), (6, 10)),
                     ((17, 1), (16, 9)),
                     ((0, 10), (12, 15)),
                     ((1, 13), (17, 18)),
                     ((12, 13), (16, 8)),
                     ((2, 13), (2, 12)),
                     ((7, 2), (18, 10)),
                     ((2, 3), (17, 16)),
                     ((19, 19), (16, 11)),
                     ((1, 18), (6, 6)),
                     ((9, 12), (2, 10)),
                     ((3, 10), (17, 18)),
                     ((7, 5), (1, 6)),
                     ((0, 2), (17, 1)),
                     ((11, 5), (11, 16)),
                     ((2, 13), (0, 0)),
                     ((17, 3), (19, 11)),
                     ((19, 17), (17, 18)),
                     ((8, 1), (7, 5)),
                     ((1, 1), (16, 1)),
                     ((12, 12), (8, 14)),
                     ((19, 2), (19, 17)),
                     ((7, 13), (19, 13)),
                     ((17, 14), (16, 5)),
                     ((1, 17), (1, 11)),
                     ((2, 6), (18, 18)),
                     ((8, 18), (15, 10)),
                     ((1, 1), (11, 14)),
                     ((1, 14), (8, 18)),
                     ((13, 6), (6, 8)),
                     ((12, 14), (3, 17)),
                     ((7, 4), (8, 12)),
                     ((3, 3), (4, 16)),
                     ((14, 4), (14, 5)),
                     ((4, 3), (13, 18)),
                     ((3, 0), (14, 1)),
                     ((3, 4), (8, 9)),
                     ((6, 15), (13, 1)),
                     ((10, 5), (3, 2)),
                     ((2, 17), (2, 16)),
                     ((2, 18), (5, 1)),
                     ((13, 8), (9, 7)),
                     ((2, 11), (7, 6)),
                     ((19, 3), (4, 4)),
                     ((4, 18), (13, 10)),
                     ((19, 3), (18, 2)),
                     ((0, 8), (3, 6)),
                     ((9, 7), (3, 7)),
                     ((13, 3), (5, 15)),
                     ((16, 1), (16, 6)),
                     ((11, 8), (7, 13)),
                     ((15, 12), (11, 13)),
                     ((8, 13), (19, 1)),
                     ((3, 9), (10, 3)),
                     ((16, 3), (0, 4)),
                     ((17, 14), (7, 19)),
                     ((10, 12), (13, 5)),
                     ((6, 0), (14, 6)),
                     ((12, 15), (4, 5)),
                     ((18, 2), (18, 7)),
                     ((6, 5), (18, 17)),
                     ((7, 6), (0, 19)),
                     ((1, 18), (7, 4)),
                     ((15, 13), (9, 8)),
                     ((17, 2), (10, 13)),
                     ((10, 17), (7, 5)),
                     ((10, 1), (0, 2)),
                     ((18, 3), (4, 18)),
                     ((4, 16), (8, 7)),
                     ((19, 13), (6, 19)),
                     ((11, 3), (0, 5)),
                     ((0, 3), (2, 3)),
                     ((1, 1), (3, 5)),
                     ((19, 11), (2, 3)),
                     ((15, 15), (12, 5)),
                     ((19, 17), (2, 14)),
                     ((2, 7), (12, 16)),
                     ((6, 5), (12, 11)),
                     ((7, 6), (5, 7)),
                     ((5, 19), (10, 1)),
                     ((3, 3), (9, 9)),
                     ((17, 19), (1, 11)),
                     ((3, 4), (3, 2)),
                     ((15, 14), (7, 13)),
                     ((4, 10), (3, 13)),
                     ((2, 4), (14, 8)),
                     ((6, 9), (8, 3)),
                     ((4, 1), (6, 14)),
                     ((18, 13), (10, 2)),
                     ((6, 17), (8, 12)),
                     ((1, 0), (12, 8)),
                     ((12, 8), (12, 16)),
                     ((6, 15), (17, 11)),
                     ((14, 11), (14, 17)),
                     ((9, 8), (12, 10)),
                     ((11, 8), (16, 19)),
                     ((2, 2), (1, 16)),
                     ((2, 10), (11, 18)),
                     ((0, 3), (19, 18)),
                     ((18, 10), (9, 18)),
                     ((11, 19), (19, 11)),
                     ((16, 2), (6, 19)),
                     ((19, 12), (6, 9)),
                     ((16, 12), (3, 3)),
                     ((4, 7), (15, 13)),
                     ((18, 15), (16, 17)),
                     ((4, 10), (7, 6)),
                     ((1, 11), (5, 14)),
                     ((13, 1), (2, 5)),
                     ((9, 19), (15, 8)),
                     ((3, 9), (7, 16)),
                     ((17, 3), (0, 15)),
                     ((14, 16), (13, 15)),
                     ((5, 17), (11, 4)),
                     ((13, 11), (1, 9)),
                     ((5, 15), (3, 8)),
                     ((5, 13), (6, 8)),
                     ((3, 11), (4, 1)),
                     ((17, 2), (1, 11))]

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
