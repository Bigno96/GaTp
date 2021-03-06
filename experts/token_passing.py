"""
Token Passing algorithm

Number of agents cooperating is defined as a function parameter
Path1, Path2 functions in the pseudocode are, respectively:
    - experts.a_star
    - TPAgent.go_to_resting_pos

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
import timeit

import experts.tp_agent as tp_ag
import experts.a_star as a_s
import utils.expert_utils as exp_utils
import numpy as np

from collections import deque
from typing import List, Tuple, Dict, Deque


def tp(input_map: np.array,
       start_pos_list: List[Tuple[int, int]],
       task_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
       parking_spot_list: List[Tuple[int, int]],
       agent_schedule: Dict[int, Deque[Tuple[int, int, int]]],
       goal_schedule: Dict[int, Deque[Tuple[int, int, int]]],
       metrics: Dict[str, List[int or float]],
       execution: exp_utils.StopToken,
       imm_task_split: float = 0.,
       new_task_per_insertion: int = 1,
       step_between_insertion: int = 1
       ) -> None:
    """
    Token Passing algorithm
    :param input_map: matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param start_pos_list: [(x,y), ...] coordinates over the map
    :param task_list: [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param parking_spot_list: (x,y) -> coordinates over the map
                              Non-task endpoints for the agent to rest on to avoid deadlocks, can be empty
    :param imm_task_split: 1 >= x >= 0, % of task_list to add to active_task
    :param new_task_per_insertion: > 0, how many 'new' task from task_list to add to active_task at each insertion
    :param step_between_insertion: > 0, how many timestep between each insertion
    :param agent_schedule, RETURN VALUE
             agent_schedule -> {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, 1), ...])
    :param goal_schedule, RETURN VALUE
             goal_schedule -> {agent_id : schedule}
                                with schedule = deque([(current_goal, 0), (curr_goal, 1), ...])
                                curr_goal -> position the agent is trying to reach
    :param metrics, RETURN VALUE
             'service_time': number of timesteps for completing each task
             'timestep_runtime': execution time of each timestep, in ms
    :param execution: used to terminate hanging instances
                      when using stop, return values behaviour is undefined
    """
    # starting positions are used as non-task endpoints
    non_task_ep_list = start_pos_list + parking_spot_list

    # precompute heuristics maps towards all endpoints
    h_coll = exp_utils.preprocess_heuristics(input_map=input_map,
                                             task_list=task_list,
                                             non_task_ep_list=non_task_ep_list)

    # instantiate agents and free agent queue
    agent_pool = {tp_ag.TpAgent(name=idx,
                                input_map=input_map,
                                start_pos=start_pos_list[idx],
                                h_coll=h_coll)
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
    split_idx = int(imm_task_split * len(task_list))
    active_task_list = task_list[:split_idx]
    new_task_pool = deque(task_list[split_idx:])

    # set up checks for how many tasks have been injected
    activated_task_count = len(active_task_list)
    total_task_count = len(task_list)

    # set up agent_schedule and goal_schedule
    for agent in agent_pool:
        agent_schedule[agent.name] = agent.path.copy()
        goal_schedule[agent.name] = agent.path.copy()
        agent.move_one_step()  # move them to timestep 1

    # track time and metrics
    timestep = 1  # timestep = 0 is the initialization
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
                metrics['service_time'].append(len(agent.path))  # add path length to complete the task

        # check for eventual collisions and adapt
        for agent in agent_pool:
            agent.collision_shielding(token=token, sys_timestep=timestep, agent_pool=agent_pool)

        # all agents move along their paths in token for one timestep
        for agent in agent_pool:
            # update agent and goal schedule
            agent_schedule[agent.name].append(agent.path[0])
            goal_schedule[agent.name].append((agent.goal[0], agent.goal[1], timestep))
            # agents update here if they are free or not (when they end a path, they become free)
            agent.move_one_step()
            # update token
            token[agent.name]['pos'] = agent.pos

        # update timings
        elapsed_time = timeit.default_timer() - start_time
        metrics['timestep_runtime'].append(elapsed_time * 1000)  # ms conversion

        timestep += 1

    # update agent and goal schedule with last move
    for agent in agent_pool:
        agent_schedule[agent.name].append(agent.path[0])
        goal_schedule[agent.name].append(agent.path[0])


# noinspection PyTypeChecker
def online_tp(input_map: np.array,
              start_pos_list: List[Tuple[int, int]],
              new_task_pool: Deque[Tuple[Tuple[int, int], Tuple[int, int]]],
              active_task_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
              assigned_task_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
              agent_schedule: Dict[int, Deque[Tuple[int, int, int]]],
              goal_schedule: Dict[int, Deque[Tuple[int, int, int]]],
              metrics: Dict[str, List[int or float]],
              execution: exp_utils.StopToken,
              timestep_limit: int = 0,
              new_task_per_insertion: int = 1,
              step_between_insertion: int = 1,
              ) -> None:
    """
    Token Passing algorithm, called during Dataset Aggregation
    :param input_map: matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param start_pos_list: [(x,y), ...] coordinates over the map
    :param new_task_pool: pool of new tasks to activate
    :param active_task_list: list of already available tasks
    :param assigned_task_list: list of tasks assigned to the agents, positional
    :param new_task_per_insertion: > 0, how many 'new' task from task_list to add to active_task at each insertion
    :param step_between_insertion: > 0, how many timestep between each insertion
    :param agent_schedule, RETURN VALUE
             agent_schedule -> {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, 1), ...])
    :param goal_schedule, RETURN VALUE
             goal_schedule -> {agent_id : schedule}
                                with schedule = deque([(current_goal, 0), (curr_goal, 1), ...])
                                curr_goal -> position the agent is trying to reach
    :param metrics, RETURN VALUE
             'service_time': number of timesteps for completing each task
             'timestep_runtime': execution time of each timestep, in ms
    :param execution: used to terminate hanging instances
                      when using stop, return values behaviour is undefined
    :param timestep_limit: for how many timestep to run the execution
                           0 means no limit
    """
    # starting positions are used as non-task endpoints
    non_task_ep_list = start_pos_list

    # precompute heuristics maps towards all endpoints
    task_list = list(new_task_pool) \
                + [task for task in assigned_task_list if task] \
                + active_task_list
    h_coll = exp_utils.preprocess_heuristics(input_map=input_map,
                                             task_list=task_list,
                                             non_task_ep_list=non_task_ep_list)

    # instantiate agents and free agent queue
    agent_pool = {tp_ag.TpAgent(name=idx,
                                input_map=input_map,
                                start_pos=start_pos_list[idx],
                                h_coll=h_coll)
                  for idx in range(len(start_pos_list))
                  }
    free_agent_queue = deque()

    # instantiate token, dict -> {agent_name : {pos: (x,y), path: [(x,y,t), ...]}}
    token = {}
    for agent in agent_pool:
        token[agent.name] = {'pos': agent.pos,
                             'path': agent.path.copy()}

    # assign agents to their initial task
    for idx, agent in enumerate(agent_pool):
        if assigned_task_list[idx]:  # if they have a task to do
            try:
                del token[agent.name]
                # first, from curr_pos to pickup_pos
                pickup_pos, delivery_pos = assigned_task_list[idx]
                pickup_path, pick_len = a_s.a_star(input_map=input_map,
                                                   start=agent.pos,
                                                   goal=pickup_pos,
                                                   token=token,
                                                   h_map=h_coll[pickup_pos],
                                                   starting_t=0,
                                                   include_start_node=True)
                # second, from pickup_pos to delivery_pos
                delivery_path, _ = a_s.a_star(input_map=input_map,
                                              start=pickup_pos,
                                              goal=delivery_pos,
                                              token=token,
                                              h_map=h_coll[delivery_pos],
                                              starting_t=pick_len,
                                              include_start_node=False)
                # merge paths and update
                agent.path = pickup_path + delivery_path
                # update goal
                agent.goal = pickup_pos
                # update agent status
                agent.is_free = False
                agent.is_idle = False

            # since MAPD can be not well-formed, it can happen to not find a path
            except exp_utils.NoPathError:
                # insert back that task to the active list
                active_task_list.append(assigned_task_list[idx])
                assigned_task_list[idx] = ()

            # update back token
            token[agent.name] = {'pos': agent.pos,
                                 'path': agent.path}

    # set up checks for how many tasks have been injected
    activated_task_count = len(active_task_list) \
                           + len([task for task in assigned_task_list if task])  # don't count None
    total_task_count = len(task_list)

    # set up agent_schedule and goal_schedule
    for agent in agent_pool:
        agent_schedule[agent.name] = deque([(agent.pos[0], agent.pos[1], 0)])
        goal_schedule[agent.name] = deque([(agent.goal[0], agent.goal[1], 0)])
        agent.move_one_step()  # move them to timestep 1

    # track time and metrics
    timestep = 1  # timestep = 0 is the initialization
    metrics['service_time'] = []
    metrics['timestep_runtime'] = []

    # while tasks are available or at least one agent is still busy
    while active_task_list or activated_task_count < total_task_count \
            or not all(ag.is_free for ag in agent_pool):

        # exit condition, avoid hanging
        # undefined return behaviour
        if execution.is_cancelled:
            return

        # online execution during DAgger has limited number of timesteps
        if timestep != 0 and timestep >= timestep_limit:
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
                metrics['service_time'].append(len(agent.path))  # add path length to complete the task

        # check for eventual collisions and adapt
        for agent in agent_pool:
            agent.collision_shielding(token=token, sys_timestep=timestep, agent_pool=agent_pool)

        # all agents move along their paths in token for one timestep
        for agent in agent_pool:
            # update agent and goal schedule
            agent_schedule[agent.name].append(agent.path[0])
            goal_schedule[agent.name].append((agent.goal[0], agent.goal[1], timestep))
            # agents update here if they are free or not (when they end a path, they become free)
            agent.move_one_step()
            # update token
            token[agent.name]['pos'] = agent.pos

        # update timings
        elapsed_time = timeit.default_timer() - start_time
        metrics['timestep_runtime'].append(elapsed_time * 1000)  # ms conversion

        timestep += 1

    # update agent and goal schedule with last move
    for agent in agent_pool:
        agent_schedule[agent.name].append(agent.path[0])
        goal_schedule[agent.name].append(agent.path[0])
