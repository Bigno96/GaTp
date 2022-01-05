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
from copy import deepcopy

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
    agent_schedule = deepcopy(token)

    # track time to add new tasks
    timestep = 1    # start from 1 since new tasks for timestep 0 are already active

    # while tasks are available or at least one agent is still busy
    while active_task_list or activated_task_count < total_task_count \
            or any(not ag.is_free for ag in agent_pool):

        # list of free agents that will request the token
        free_agent_queue = deque([agent
                                  for agent in agent_pool
                                  if agent.is_free])

        # while agent a_i exists that requests token
        while free_agent_queue:
            agent = free_agent_queue.pop()
            # pass control to agent a_i
            agent.receive_token(token=token,
                                task_list=active_task_list,
                                non_task_ep_list=non_task_ep_list)
            # a_i has updated token, active_task_list and its 'free' status

        # all agents move along their paths in token for one timestep
        for agent in agent_pool:
            # agents update also here if they are free or not (when they end a path, they become free)
            agent.move_one_step()
            # update schedule
            agent_schedule[agent.name].append(agent.path[0])

        # add new tasks, if any, before next iteration
        if (timestep % step_between_insertion) == 0:    # every n step add new tasks
            new_task_list = [new_task_pool.popleft()
                             for _ in range(min(len(new_task_pool), new_task_per_insertion))
                             ]
            active_task_list.extend(new_task_list)
            activated_task_count += len(new_task_list)
        timestep += 1

    return agent_schedule
