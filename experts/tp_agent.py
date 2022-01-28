"""
Token Passing Agent class for TP Algorithm

Define agent's move, find resting position and receive token
"""


from collections import deque

import numpy as np

from experts.a_star import a_star
from utils.expert_utils import MOVE_LIST, free_cell_heuristic


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
        self.map = input_map
        self.pos = start_pos    # tuple (x,y)
        self.h_coll = h_coll
        # path the agent is following, [(x_0, y_0, t_0), (x_1, y_1, t_1), ..., (x_g, y_g, t_g)]
        self.path = deque([(start_pos[0], start_pos[1], 0)])
        self.is_free = True

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return (f'TpAgent(name={self.name}, free={self.is_free}, pos={self.pos}\n'
                f'\t\tpath={self.path})')

    def __hash__(self):
        return hash(self.name)

    def move_one_step(self):
        """
        Agent moves one step down its path
        By assumption, agents stay in place after finishing their tasks (= at the end of their paths)
        """
        # if last step
        if len(self.path) == 1:
            self.pos = self.path[-1][:-1]
            self.path[-1] = (self.pos[0], self.pos[1], self.path[-1][-1]+1)
            self.is_free = True
        else:
            self.pos = self.path.popleft()[:-1]  # move

    def find_resting_pos(self, token, task_list, non_task_ep_list, sys_timestep):
        """
        Pick the nearest endpoint s.t. delivery locations of all tasks are different from the chosen endpoint,
        no path of other agents in the token ends in the chosen endpoint
        and does not collide with the paths of other agents stored in the token
        Update its path with minimal cost path to the chosen endpoint
        :param token: summary of other agents planned paths
                      dict -> {agent_id : path}
                      with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...])
                      x, y -> cartesian coords, t -> timestep
        :param task_list: list of tasks -> [(task1), (task2), ...]
                          task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
        :param non_task_ep_list: list of endpoints not belonging to a task -> [(ep1), (ep2), ...]
                                 endpoint: tuple (x,y) of int coordinates
        :param sys_timestep: global timestep of the execution
        """
        # task related endpoints, excluding all delivery locations in task_list
        # -> get only pickup locations
        ep_list = [pickup
                   for pickup, _ in task_list]
        # list of delivery spots, needs to check since can happen that pickup == delivery for different tasks
        del_list = [delivery
                    for _, delivery in task_list]
        ep_list = list(set(ep_list) - set(del_list))
        # add non task related endpoints
        ep_list.extend(non_task_ep_list)

        # get list of all endpoints in the token (cutting off timesteps)
        token_ep_list = [path[-1][:-1]
                         for path in token.values()]
        # remove an endpoint if it's an endpoint also for another agent's path in the token
        ep_list = list(set(ep_list) - set(token_ep_list))

        # sort based off heuristic value, ascending order (lowest first)
        sorted_ep_list = deque(sorted(ep_list, key=lambda ep: self.h_coll[ep][self.pos]))

        # while there are still endpoints to try
        while sorted_ep_list:
            # get best endpoint
            best_ep = sorted_ep_list.popleft()

            try:
                # collision free path, if endpoint is reachable
                self.path, _ = a_star(input_map=self.map, start=self.pos, goal=best_ep,
                                      token=token, h_map=self.h_coll[best_ep], starting_t=sys_timestep)
                token[self.name] = self.path
                self.is_free = True

                return

            except ValueError:
                pass  # keep going and try another endpoint

        # no endpoint was reachable -> stay in place
        # this happens due to MAPD instance not being well-formed
        token[self.name] = self.path
        self.is_free = True

    def receive_token(self, token, task_list, non_task_ep_list, sys_timestep):
        """
        Agent receives token and assigns himself to a new task
        Add its new path to the token, remove assigned task from task_list
        :param token: summary of other agents planned paths
                      dict -> {agent_id : path}
                      with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...])
                      x, y -> cartesian coords, t -> timestep
        :param task_list: list of tasks -> [(task1), (task2), ...]
                          task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
        :param non_task_ep_list: list of endpoints not belonging to a task -> [(ep1), (ep2), ...]
                                 endpoint: tuple (x,y) of int coordinates
        :param sys_timestep: global timestep of the execution
        :return: assigned task, None if no task assigned
        """

        # remove himself from the token
        # most functions assume that current agent path is not in the token (since it's to be decided)
        del token[self.name]

        # get subset of tasks s.t. their pickup or delivery spots don't coincide with endpoints of token paths
        avail_task_list = [task for task in task_list
                           # check paths endpoints in token (only coordinates, removing timestep)
                           if not any(loc in [path[-1][:-1] for path in token.values()]
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
                pickup_path, pick_len = a_star(input_map=self.map,
                                               start=self.pos, goal=pickup_pos,
                                               token=token,
                                               h_map=self.h_coll[pickup_pos],
                                               starting_t=sys_timestep)
                # second, from pickup_pos to delivery_pos
                delivery_path, _ = a_star(input_map=self.map,
                                          start=pickup_pos, goal=delivery_pos,
                                          token=token,
                                          h_map=self.h_coll[delivery_pos],
                                          starting_t=sys_timestep+pick_len-1)
                # remove leftmost element since pickup path already ends there
                delivery_path.popleft()
                # merge paths and update
                self.path = pickup_path + delivery_path

                # assign task
                task_list.remove(best_task)
                # update token
                token[self.name] = self.path
                self.is_free = False

                return best_task

            # since MAPD can be not well-formed, it can happen to not find a path
            except ValueError:
                token[self.name] = self.path    # try another timestep
                self.is_free = True

                return None

        # no task in task_list has delivery_pos == self.pos
        elif all([delivery != self.pos for _, delivery in task_list]):
            # stay in place
            token[self.name] = self.path
            self.is_free = True

        # no available task, agent is in a delivery spot
        else:
            # move to another reachable endpoint
            self.find_resting_pos(token=token, task_list=task_list, non_task_ep_list=non_task_ep_list,
                                  sys_timestep=sys_timestep)

        return None

    def collision_shielding(self, token, sys_timestep, agent_pool):
        """
        Avoid collisions by moving an agent if another one is coming into its resting spot
        :param token: summary of other agents planned paths
                      dict -> {agent_id : path}
                      with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...])
                      x, y -> cartesian coords, t -> timestep
        :param sys_timestep: global timestep of the execution
        :param agent_pool: set of agents
        """
        # if the agent is parked in a resting spot
        if len(self.path) == 1:

            # remove himself from the token, if present
            # most functions assume that current agent path is not in the token (since it's to be decided)
            del token[self.name]

            # if some other agent is coming into agent end path position on the next timestep
            end_pos = self.path[-1][:-1]
            if end_pos in {(x, y)
                           for path in token.values()
                           for x, y, t in path
                           if t == sys_timestep+1}:

                # try to move the agent towards a non-conflicting cell around him
                d1_cell_list = [(end_pos[0]+move[0], end_pos[1]+move[1])  # distance 1
                                for move in MOVE_LIST]
                # reverse order, higher number of free cells first
                # count free cell at sys_timestep+1 when the agent will be in target
                d1_cell_list = sorted(d1_cell_list, reverse=True,
                                      key=lambda c: free_cell_heuristic(target=c, input_map=self.map,
                                                                        token=token,
                                                                        target_timestep=sys_timestep+1))

                # loop over d1 cells
                for cell in d1_cell_list:
                    try:
                        self.path, _ = a_star(input_map=self.map, start=end_pos, goal=cell,
                                              token=token, h_map=None,     # cell is not always an endpoint
                                              starting_t=sys_timestep)

                        # if agent is going to 'disturb' another agent, call collision shielding on him
                        # agent can be disturbed only if standing still
                        # otherwise -> node collision prevention will avoid the conflict
                        disturbed_agent_names = [ag
                                                 for ag, path in token.items()
                                                 for x, y, t in path
                                                 if t == sys_timestep    # next timestep
                                                 and (x, y) == self.path[-1][:-1]  # potential collision incoming
                                                 ]

                        # add here path to token for others
                        token[self.name] = self.path
                        self.is_free = True

                        for ag in disturbed_agent_names:
                            # get the agent
                            agent = [a
                                     for a in agent_pool
                                     if a.name == ag][0]
                            # run coll shield
                            agent.collision_shielding(token=token, sys_timestep=sys_timestep,
                                                      agent_pool=agent_pool)
                        # all agent called, return
                        return

                    # no path, try another cell
                    except ValueError:
                        pass

            # re add agent path to token if nothing done
            token[self.name] = self.path
