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

    # update schedule with last move
    for agent in agent_pool:
        agent_schedule[agent.name].append(agent.path[0])
