"""
Token Passing algorithm

Number of agents cooperating is defined as a function parameter
Path1, Path2 functions in the pseudocode are, respectively:
    - experts.a_star
    - experts.funcs.find_resting_pos

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
import collections
from copy import deepcopy

import numpy as np

from experts.a_star import a_star
from experts.funcs import find_resting_pos, preprocess_heuristics


class TpAgent:
    """
    Class for Agents in Token Passing algorithms
    """

    def __init__(self, name, input_map, start_pos, h_coll):
        """
        Initialize the class
        :param name: int, unique identifier
        :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
        :param start_pos: tuple (x,y), absolute position of the agent in the map
        :param h_coll: dict -> {endpoint : h_map}
                       endpoint: tuple (x,y) of int coordinates
                       h_map: np.ndarray, type=int, shape=input_map.shape, heuristic matrices with goal = endpoint
        """
        self.name = name
        self.__map = input_map
        self.pos = start_pos    # tuple (x,y)
        self.__h_coll = h_coll
        # path the agent is following, [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
        self.path = [(self.pos(0), self.pos(1), 0)]
        self.free = True

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return (f'TpAgent(name={self.name}, free={self.free}, pos={self.pos},'
                f'\t\tpath={self.path})')

    def move_one_step(self):
        """
        Agent moves one step down its path
        By assumption, agents stay in place after finishing their tasks (= at the end of their paths)
        """
        self.pos = self.path.pop(0)[:-1]  # move
        # last step was taken
        if not self.path:
            self.path = [(self.pos[0], self.pos[1], 0)]     # stay in place
            self.free = True

    def receive_token(self, token, task_list, non_task_ep_list):
        """
        Agent receives token and assigns himself to a new task
        Add its new path to the token, remove assigned task from task_list
        :param token: summary of other agents planned paths
                      dict -> {agent_id : path}
                      with path = [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
                      x, y -> cartesian coords, t -> timestep
        :param task_list: list of tasks -> [(task1), (task2), ...]
                          task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
        :param non_task_ep_list: list of endpoints not belonging to a task -> [(ep1), (ep2), ...]
                                 endpoint: tuple (x,y) of int coordinates
        :return: selected path: [(x_0, y_0, t_0), (x_1, y_1, t_1), ...]
        """
        # get subset of tasks s.t. their pickup or delivery spots don't coincide with endpoints of token paths
        avail_task_list = [task for task in task_list
                           # check paths endpoints in token (only coordinates, removing timestep)
                           if not any(loc in [path[-1][:-1] for path in token.values()]
                                      # neither pickup nor delivery pos for the task are in there
                                      for loc in task)
                           ]

        # remove himself from the token, if present
        # most functions assume that current agent path is not in the token (since it's to be decided)
        if self.name in set(token.keys()):
            del token[self.name]

        # if at least one task is available
        if avail_task_list:
            # list -> [h-value1, h-value2, ...]
            # h-value from current agent position to pickup_pos of the task
            # argmin -> index of avail_task_list where task has min(h-value)
            best_task = avail_task_list[np.argmin([self.__h_coll[pickup][self.pos]
                                                   for pickup, _ in avail_task_list
                                                   ])]

            try:
                # first, curr_pos -> pickup_pos
                pickup_path, pick_len = a_star(input_map=self.__map,
                                               start=self.pos, goal=best_task[0],
                                               token=token,
                                               h_map=self.__h_coll[best_task[0]])
                # second, pickup_pos -> delivery_pos
                delivery_path, _ = a_star(input_map=self.__map,
                                          start=best_task[0], goal=best_task[1],
                                          token=token,
                                          h_map=self.__h_coll[best_task[1]])
                # adjust timesteps of second half of the path
                delivery_path = [(x, y, pick_len+t)
                                 for x, y, t in delivery_path]
                # update path
                self.path = pickup_path
                self.path.extend(delivery_path)

                # update token
                token[self.name] = self.path
                self.free = False

            # since MAPD can be not well-formed, it can happen to not find a path
            except ValueError:
                self.path = [(self.pos[0], self.pos[1], 0)]      # stay in place and try another timestep
                token[self.name] = self.path
                self.free = True

        # no task in task_list has delivery_pos == self.pos
        elif all([delivery != self.pos for delivery, _ in task_list]):
            # stay in place
            self.path = [(self.pos[0], self.pos[1], 0)]
            token[self.name] = self.path
            self.free = True

        # no available task, agent is a delivery spot
        else:
            # move to another reachable endpoint
            self.path = find_resting_pos(start=self.pos, input_map=self.__map,
                                         token=token, h_coll=self.__h_coll,
                                         task_list=task_list,
                                         non_task_ep_list=non_task_ep_list)
            token[self.name] = self.path
            self.free = True

        return self.path.copy()


def tp(input_map, start_pos_list, task_list,
       parking_spot=None, imm_task_split=0.5, new_task_per_timestep=10):
    """
    Token Passing algorithm
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param start_pos_list: list of tuples, (x,y) -> coordinates over the map
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param parking_spot: list of tuples, (x,y) -> coordinates over the map
           Optional, non-task endpoints for the agent to rest on to avoid deadlocks
    :param imm_task_split: float, 1 > x > 0, % of task_list to add to active_task
    :param new_task_per_timestep: int, > 0, how many 'new' task from task_list to add to active_task at each timestep
    :return: agent_schedule
             agent_schedule -> {agent_id : schedule}
                                with schedule = [(x_0, y_0, 0), (x_1, y_1, t_1), ...]
    """
    # perform union removing eventual repetitions
    # starting positions are used as non-task endpoints
    non_task_ep_list = list(set(start_pos_list) | set(parking_spot))

    # preprocess heuristics towards endpoints
    h_coll = preprocess_heuristics(input_map=input_map,
                                   task_list=task_list, non_task_ep_list=non_task_ep_list)

    # instantiate agents
    agent_list = [TpAgent(name=idx, input_map=input_map,
                          start_pos=start_pos_list[idx], h_coll=h_coll)
                  for idx in range(len(start_pos_list))
                  ]
    agent_name_list = [agent.name for agent in agent_list]

    # instantiate token, dict -> {agent_id : path}
    agent_trivial_paths = [[(x, y, 0)]
                           for agent in agent_list
                           for x, y in agent.pos]
    token = dict(zip(agent_name_list, iter(agent_trivial_paths)))

    # set up a list of active, immediately available tasks and a pool of 'new' tasks
    # split done according to imm_task_split value
    split_idx = int(imm_task_split*len(task_list))
    active_task_list = task_list[:split_idx]
    new_task_pool = collections.deque(task_list[split_idx:])

    # set up agent_schedule
    agent_schedule = deepcopy(token)

    # while tasks are available
    while active_task_list:

        # list of free agents that will request the token
        free_agent_queue = collections.deque([agent
                                              for agent in agent_list
                                              if agent.free])

        # while agent a_i exists that requests token
        while free_agent_queue:
            agent = free_agent_queue.pop()
            # pass control to agent a_i
            path = agent.receive_token(token=token,
                                       task_list=active_task_list,
                                       non_task_ep_list=non_task_ep_list)
            # a_i has updated token, active_task_list and its 'free' status
            # update schedule
            agent_schedule[agent.name].extend(path)

        # all agents move along their paths in token for one timestep
        for agent in agent_list:
            # agents update also here if they are free or not (when they end a path, they become free)
            agent.move_one_step()

        # add new tasks, if any, before next iteration
        active_task_list.extend([new_task_pool.pop()
                                 for _ in range(min(len(new_task_pool), new_task_per_timestep))
                                 ])

    return agent_schedule
