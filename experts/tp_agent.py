"""
Token Passing Agent class for TP Algorithm

Define agent's move, find resting position and receive token
"""
from __future__ import annotations

import numpy as np
import experts.a_star as a_s
import utils.expert_utils as exp_utils

from collections import deque
from typing import Optional, Dict, List, Tuple, Deque


class TpAgent:
    """
    Class for Agents in Token Passing algorithms
    """

    def __init__(self,
                 name: int,
                 input_map: np.array,
                 start_pos: Tuple[int, int],
                 h_coll: Dict[Tuple[int, int], np.array]):
        """
        :param name: unique identifier
        :param input_map: matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
        :param start_pos: (x,y), absolute position of the agent in the map
        :param h_coll: dict -> {endpoint : h_map}
                       endpoint: (x,y), endpoint coordinates
                       h_map: shape = input_map.shape, heuristic matrices with goal = endpoint
        """
        self.name = name
        self.map = input_map
        self.pos = start_pos    # tuple (x,y)
        self.h_coll = h_coll
        # path the agent is following, [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
        self.path = deque([(start_pos[0], start_pos[1], 0)])
        # position the agent is trying to reach
        self.goal = self.pos
        self.is_free = True     # free -> agent is not doing any task (can be standing still or moving to rest)
        self.is_idle = True     # idle -> agent is free AND standing still

    def __eq__(self, other: TpAgent) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return (f'TpAgent(name={self.name}, free={self.is_free}, idle={self.is_idle}, pos={self.pos}\n'
                f'\t\tpath={self.path})')

    def __hash__(self) -> int:
        return hash(self.name)

    def move_one_step(self) -> None:
        """
        Agent moves one step down its path
        By assumption, agents stay in place at the end of their paths
        """
        # if last step
        if len(self.path) == 1:
            self.pos = self.path[-1][:-1]
            self.path[-1] = (self.pos[0], self.pos[1], self.path[-1][-1]+1)
            self.goal = self.pos
            self.is_free = True
            self.is_idle = True
        else:
            self.pos = self.path.popleft()[:-1]  # move
            # check pickup pos is reached
            if self.pos == self.goal:
                self.goal = self.path[-1][:-1]  # update with delivery position

    def receive_token(self,
                      token: Dict[int, Dict[str, Tuple[int, int] or Deque[Tuple[int, int, int]]]],
                      task_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                      non_task_ep_list: List[Tuple[int, int]],
                      sys_timestep: int
                      ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Agent receives token and assigns himself to a new task
        Add its new path to the token, remove assigned task from task_list
        :param token: summary of other agents planned paths
                  dict -> {agent_name : {'pos': (x,y), ''path': path}}
                  with pos = current agent pos
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...]), future steps
                  x, y -> cartesian coords, t -> timestep
        :param task_list: [(task1), (task2), ...]
                          task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
        :param non_task_ep_list: endpoints not belonging to a task -> [(ep1), (ep2), ...]
                                 endpoint: tuple (x,y) of int coordinates
        :param sys_timestep: global timestep of the execution
        :return: assigned task,
                 None if no task assigned
        """

        # remove himself from the token
        # most functions assume that current agent path is not in the token (since it's to be decided)
        del token[self.name]

        # get subset of tasks s.t. their pickup or delivery spots don't coincide with endpoints of token paths
        token_ep_list = [val['path'][-1][:-1] for val in token.values()]
        avail_task_list = [task for task in task_list
                           # check paths endpoints in token
                           if not any(loc in token_ep_list
                                      # neither pickup nor delivery pos for the task are in there
                                      for loc in task)
                           ]

        # if at least one task is available
        if avail_task_list:
            # list -> [h-value1, h-value2, ...]
            # h-value from current agent position to pickup_pos of the task
            # argmin -> index of avail_task_list where task has min(h-value)
            best_task = avail_task_list[np.argmin([self.h_coll[pickup][self.pos]
                                                   for pickup, _ in avail_task_list
                                                   ])]

            try:
                # first, from curr_pos to pickup_pos
                pickup_pos, delivery_pos = best_task
                pickup_path, pick_len = a_s.a_star(input_map=self.map,
                                                   start=self.pos,
                                                   goal=pickup_pos,
                                                   token=token,
                                                   h_map=self.h_coll[pickup_pos],
                                                   starting_t=sys_timestep,
                                                   include_start_node=False)
                # second, from pickup_pos to delivery_pos
                delivery_path, _ = a_s.a_star(input_map=self.map,
                                              start=pickup_pos,
                                              goal=delivery_pos,
                                              token=token,
                                              h_map=self.h_coll[delivery_pos],
                                              starting_t=sys_timestep+pick_len,
                                              include_start_node=False)
                # merge paths and update
                self.path = pickup_path + delivery_path
                # update goal
                self.goal = pickup_pos

                # assign task
                task_list.remove(best_task)
                # update token
                token[self.name] = {'pos': self.pos,
                                    'path': self.path}
                self.is_free = False
                self.is_idle = False

                return best_task

            # since MAPD can be not well-formed, it can happen to not find a path
            except exp_utils.NoPathError:
                token[self.name] = {'pos': self.pos,
                                    'path': self.path}   # try another timestep
                return None

        # no task in task_list has delivery_pos == self.pos
        elif all([delivery != self.pos for _, delivery in task_list]):
            # keep the same path
            token[self.name] = {'pos': self.pos,
                                'path': self.path}
            return None

        # no available task, agent is in a delivery spot
        else:
            # move to another reachable endpoint
            self.go_to_resting_pos(token=token,
                                   task_list=task_list,
                                   non_task_ep_list=non_task_ep_list,
                                   sys_timestep=sys_timestep)
            return None

    def go_to_resting_pos(self,
                          token: Dict[int, Dict[str, Tuple[int, int] or Deque[Tuple[int, int, int]]]],
                          task_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                          non_task_ep_list: List[Tuple[int, int]],
                          sys_timestep: int) -> None:
        """
        Pick the nearest endpoint s.t. delivery locations of all tasks are different from the chosen endpoint,
        no path of other agents in the token ends in the chosen endpoint
        and does not collide with the paths of other agents stored in the token
        Update its path with minimal cost path to the chosen endpoint
        :param token: summary of other agents planned paths
                  dict -> {agent_name : {'pos': (x,y), ''path': path}}
                  with pos = current agent pos
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...]), future steps
                  x, y -> cartesian coords, t -> timestep
        :param task_list: [(task1), (task2), ...]
                          task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
        :param non_task_ep_list: endpoints not belonging to a task -> [(ep1), (ep2), ...]
                                 endpoint: tuple (x,y) of int coordinates
        :param sys_timestep: global timestep of the execution
        """
        # task related endpoints, excluding all delivery locations in task_list
        # list of delivery spots, needs to check since can happen that pickup == delivery for different tasks
        del_list = {delivery for _, delivery in task_list}
        # -> get only pickup locations
        ep_list = [pickup for pickup, _ in task_list
                   if pickup not in del_list]
        # add non task related endpoints
        ep_list.extend(non_task_ep_list)

        # get list of all endpoints in the token (cutting off timesteps)
        token_ep_list = {val['path'][-1][:-1]
                         for val in token.values()}
        # remove an endpoint if it's an endpoint also for another agent's path in the token
        ep_list = list(set(ep_list) - token_ep_list)

        # sort based off heuristic value, ascending order (lowest first)
        sorted_ep_list = deque(sorted(ep_list,
                                      key=lambda ep: self.h_coll[ep][self.pos]))

        # while there are still endpoints to try
        while sorted_ep_list:
            # get best endpoint
            best_ep = sorted_ep_list.popleft()

            try:
                # collision free path, if endpoint is reachable
                self.path, _ = a_s.a_star(input_map=self.map,
                                          start=self.pos,
                                          goal=best_ep,
                                          token=token,
                                          h_map=self.h_coll[best_ep],
                                          starting_t=sys_timestep,
                                          include_start_node=False)
                token[self.name] = {'pos': self.pos,
                                    'path': self.path}
                self.goal = best_ep
                self.is_free = True
                self.is_idle = False

                return

            except exp_utils.NoPathError:
                pass  # keep going and try another endpoint

        # no endpoint was reachable -> keep current path
        # this happens due to MAPD instance not being well-formed
        token[self.name] = {'pos': self.pos,
                            'path': self.path}

    def collision_shielding(self,
                            token: Dict[int, Dict[str, Tuple[int, int] or Deque[Tuple[int, int, int]]]],
                            sys_timestep: int,
                            agent_pool: set[TpAgent],
                            _time_horizon: int = 3
                            ) -> None:
        """
        Avoid collisions by moving an agent if another one is coming into its current idle spot
        Scan token looking for potential future conflicts beneath the time horizon
        :param token: summary of other agents planned paths
                  dict -> {agent_name : {'pos': (x,y), ''path': path}}
                  with pos = current agent pos
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...]), future steps
                  x, y -> cartesian coords, t -> timestep
        :param sys_timestep: global timestep of the execution
        :param agent_pool: set of agents
        :param _time_horizon: maximum time distance the agent will look in the future
                              (Default: 3)
        """
        # if the agent is doing nothing
        if self.is_idle:

            # remove himself from the token, if present
            # most functions assume that current agent path is not in the token
            del token[self.name]

            # check whether the idle agent is potentially causing a conflict
            idle_pos = self.path[-1][:-1]
            next_pos_pool = {(x, y)
                             for val in token.values()
                             for x, y, t in val['path']
                             # +1 since range exclude second value
                             if t in range(sys_timestep, sys_timestep+_time_horizon+1)}

            # it is not causing any conflict
            if idle_pos not in next_pos_pool:
                # re add agent path to token
                token[self.name] = {'pos': self.pos,
                                    'path': self.path}
                return

            # some other agent is coming into agent end path position on this timestep -> conflict

            # don't consider idle agents -> make them re-plan after
            idle_token = exp_utils.drop_idle(agent_pool=agent_pool, curr_agent=self, token=token)

            # try to move the agent towards a non-conflicting cell around him
            d1_cell_list = [(idle_pos[0]+move[0], idle_pos[1]+move[1])  # distance 1
                            for move in exp_utils.NEIGHBOUR_LIST]
            # reverse order, higher number of free cells first
            # count free cell at sys_timestep when the agent will be in target
            d1_cell_list = sorted(d1_cell_list, reverse=True,
                                  key=lambda c: exp_utils.free_cell_heuristic(target=c,
                                                                              input_map=self.map,
                                                                              token=token,
                                                                              target_timestep=sys_timestep))

            # loop over d1 cells
            for cell in d1_cell_list:
                try:
                    self.path, _ = a_s.a_star(input_map=self.map, start=idle_pos, goal=cell,
                                              token=token, h_map=None,     # cell is not always an endpoint
                                              starting_t=sys_timestep,
                                              include_start_node=False)

                    # re add removed idle
                    token.update(idle_token)

                    # if agent is going to 'disturb' another agent, call collision shielding on him
                    # agent can be disturbed only if they are idle
                    # otherwise -> node collision prevention will avoid the conflict
                    disturbed_agent_names = [ag
                                             for ag, val in token.items()
                                             for x, y, t in val['path']
                                             if t == sys_timestep    # next move
                                             and (x, y) == self.path[0][:-1]  # potential collision incoming
                                             ]

                    # add here path to token for others CS
                    token[self.name] = {'pos': self.pos,
                                        'path': self.path}
                    self.goal = cell
                    self.is_free = True
                    self.is_idle = False

                    for ag in disturbed_agent_names:
                        # get the agent
                        agent = [a
                                 for a in agent_pool
                                 if a.name == ag][0]
                        # run coll shield
                        agent.collision_shielding(token=token, sys_timestep=sys_timestep,
                                                  agent_pool=agent_pool)
                    # all disturbed agents called, return
                    return

                # no path, try another cell
                except exp_utils.NoPathError:
                    pass

            # impossible to avoid collision
            # re add agent path to token
            token[self.name] = {'pos': self.pos,
                                'path': self.path}
            # re add removed idle
            token.update(idle_token)
