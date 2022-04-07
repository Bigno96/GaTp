"""
Class for handling Multi Agent Execution Simulation

Supported ops:
    - Keep track of all agents position
    - Handle task assignment
    - Move agents following learned Action Policy
"""

import torch
import logging

import numpy as np
import utils.agent_state as ag_state
import utils.graph_utils as g_utils
import utils.expert_utils as exp_utils

from itertools import repeat
from collections import deque
from easydict import EasyDict
from typing import Dict, List, Tuple, Deque


class MultiAgentSimulator:
    """
    Supporting operations for simulation
    Used to simulate a solution of a MAPD problem during validation/testing

    Agents will move accordingly to learned action policy
    Task assignment is centralized
    """

    def __init__(self,
                 config: EasyDict,
                 device: str):
        """
        :param config: configuration Namespace
        :param device: device on which to execute computation
        """
        self.config = config

        '''basic'''
        self.agent_num = self.config.agent_number
        # NN trained model
        self.model = torch.nn.Module()
        # agent state representation
        self.agent_state = ag_state.AgentState(config=self.config)
        # obstacle map
        self.map = np.array(())     # H x W

        '''simulation data structures'''
        self.agent_start_pos = np.array(())     # agent_num x 2
        self.curr_agent_pos = np.array(())  # agent_num x 2

        # keeping track of task assignment, { id : shape = (2x2 or 1x2 or ()) }
        self.task_register: Dict[int, np.array] = {}
        self.task_list = np.array(())   # task_num x 2 x 2
        self.active_task_list: List[np.array] = []  # tracking active tasks
        self.new_task_pool: Deque[np.array] = deque()  # new tasks still to activate

        # keeping track of agent movements
        self.agent_schedule: Dict[int, Deque[Tuple[int, int, int]]] = {}
        # precomputed distance heuristics
        self.h_coll: Dict[Tuple, np.array] = {}

        '''simulation parameters'''
        self.timestep = 0  # timestep counter
        self.terminate = False  # terminate boolean
        self.max_step_factor: int = self.config.max_step_factor  # maximum step factor allowed in the simulation

        self.task_number: int = self.config.task_number  # total number of tasks
        self.activated_task_count = 0  # number of tasks activated

        self.split_idx = int(self.config.imm_task_split * self.task_number)  # % of task to immediately activate
        self.new_task_per_timestep: int = self.config.new_task_per_timestep  # tasks to add at each timestep
        self.step_between_insertion: int = self.config.step_between_insertion  # timestep between each insertion

        # device
        self.device = device

        '''pre-define directions'''
        self.up = np.array([-1, 0], dtype=np.int8)
        self.down = np.array([1, 0], dtype=np.int8)
        self.left = np.array([0, -1], dtype=np.int8)
        self.right = np.array([0, 1], dtype=np.int8)
        self.stop = np.array([0, 0], dtype=np.int8)
        self.up_idx = 0  # indexes for action selection out of model prediction
        self.left_idx = 1
        self.down_idx = 2
        self.right_idx = 3
        self.stop_idx = 4

    def simulate(self,
                 obstacle_map: torch.Tensor,
                 start_pos_list: torch.Tensor,
                 task_list: torch.Tensor,
                 model: torch.nn.Module,
                 target_makespan: int,
                 ) -> None:
        """
        :param obstacle_map: shape = (H, W)
        :param start_pos_list: shape = (agent_num, 2)
        :param task_list: shape = (task_num, 2, 2)
        :param model: trained model
        :param target_makespan: makespan of the expert solution
        """
        # maximum step allowed for the simulation
        max_step = int(target_makespan * self.max_step_factor)
        # set up simulator
        self.set_up_simulation(obstacle_map=obstacle_map,
                               ag_start_pos=start_pos_list,
                               task_list=task_list,
                               model=model)

        # loop until termination or max step is reached
        while not self.terminate and self.timestep < (max_step-1):   # -1 since timestep update is at the start
            self.execute_one_timestep()

    def set_up_simulation(self,
                          obstacle_map: torch.Tensor,
                          ag_start_pos: torch.Tensor,
                          task_list: torch.Tensor,
                          model: torch.nn.Module,
                          ) -> None:
        """
        Set up variables for the simulation
        :param obstacle_map: shape = (H, W)
        :param ag_start_pos: shape = (agent_num, 2)
        :param task_list: shape = (task_num, 2, 2)
        :param model: trained model
        """
        # fix model for evaluation mode
        self.model = model
        self.model.eval()

        # init map and set it for agent state
        self.map = obstacle_map.detach().cpu().numpy().astype(np.int8)  # convert it to numpy
        self.agent_state.set_obstacle_map(input_map=self.map)

        # scalar variables init
        self.terminate = False
        self.timestep = 0  # time 0 is for setup
        self.activated_task_count = self.split_idx  # update activated count with immediate tasks

        # agent start and current position
        self.agent_start_pos = ag_start_pos.detach().cpu().numpy().astype(np.int8)  # convert it to numpy
        self.curr_agent_pos = self.agent_start_pos.copy()  # copy to avoid memory interference

        # agent schedule init with agent start pos
        for i in range(self.agent_num):
            self.agent_schedule[i] = deque([(self.agent_start_pos[i, 0],
                                             self.agent_start_pos[i, 1],
                                             0)])  # timestep
        # init goal register as empty
        self.task_register = dict(zip(range(self.agent_num), repeat(np.array(()))))

        # task init
        self.task_list = task_list.detach().cpu().numpy().astype(np.int8)  # convert it to numpy
        self.active_task_list = list(self.task_list[:self.split_idx])  # activate first n tasks
        self.new_task_pool = deque(self.task_list[self.split_idx:])  # put the rest in the waiting pool

        # precompute distance heuristics towards all possible endpoints
        self.h_coll = exp_utils.preprocess_heuristics(input_map=self.map,
                                                      task_list=self.task_list,
                                                      # agents don't use parking positions
                                                      non_task_ep_list=[])

    def execute_one_timestep(self) -> None:
        """
        Execute one timestep of the simulation
        """
        # update timestep
        self.timestep += 1

        # every n step add new tasks
        if (self.timestep % self.step_between_insertion) == 0:
            new_task_list = [self.new_task_pool.popleft()
                             for _ in range(min(len(self.new_task_pool), self.new_task_per_timestep))
                             ]
            self.active_task_list.extend(new_task_list)
            self.activated_task_count += len(new_task_list)

        # handle tasks for all agents
        self.update_task_register()
        # get a goal for every agent
        goal_list = self.get_goal_list()

        # get an input for the model
        # np array -> (num_agents, 3 (channels), FOV+2*border, FOV+2*border)
        input_tensor = self.agent_state.get_input_state(goal_pos_list=goal_list,
                                                        agent_pos_list=self.curr_agent_pos)
        # shape = 1 x N x 3 x F_H x F_W
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device).float()   # add batch = 1

        # obtain and set gso
        GSO = g_utils.compute_adj_matrix(agent_pos_list=self.curr_agent_pos,
                                         comm_radius=self.config.comm_radius)
        GSO = torch.from_numpy(GSO).unsqueeze(0).to(self.device).float()     # add batch = 1
        self.model.set_gso(GSO)

        # predict with model
        model_output = self.model(input_tensor)  # B*N x 5 -> since B=1, N x 5

        try:
            # exp_multinorm for getting predicted action
            action_idx_predict = self.exp_multinorm(model_output)  # 1*N x 1 (since B=1)

            # move agents
            self.move_agents(action_idx_predict)

            # check end condition
            if (self.activated_task_count == self.task_number  # all tasks have been added
                    and not self.active_task_list  # all tasks have been assigned
                    and not any(
                        arr.size for arr in self.task_register.values())):  # all agents have finished their task
                self.terminate = True

        except Exception as err:
            logger = logging.getLogger('Agent')
            torch.set_printoptions(threshold=100000)

            logger.warning(err)
            logger.warning(f'Printing state of variables causing the error\n')
            logger.warning(f'AGENT POSITIONS: {self.curr_agent_pos}')
            logger.warning(f'INPUT TENSOR')
            logger.warning(f'Obstacle channel of input Tensor:\n {input_tensor[:,:,0,:,:]}')
            logger.warning(f'Goal channel of input Tensor:\n {input_tensor[:, :, 1, :, :]}')
            logger.warning(f'Agent position channel of input Tensor:\n {input_tensor[:, :, 2, :, :]}')
            logger.warning(f'GSO: {GSO}')
            logger.warning(f'MODEL OUTPUT: {model_output}')

            exit(-1)

    def update_task_register(self) -> None:
        """
        Check how agents are doing with their tasks
        1) If an agent has no task -> assign a new one, if possible
        2a) If an agent has reached a pickup position -> give him the delivery position
        2b) If an agent has reached a delivery position -> assign a new task, if possible
        3) If an agent has a task and has not reached any endpoint -> do nothing
        """
        # loop over all agents
        for i in range(self.agent_num):

            curr_task = self.task_register[i].copy()  # get agent current task (ndarray, shape=2x2 or 1x2 or ())
            agent_pos = self.curr_agent_pos[i]  # get agent current pos (ndarray, shape=2)

            # case 1: no task assigned
            if not curr_task.size:
                curr_task = self.assign_closest_task(agent_pos=agent_pos)  # find task (ndarray, shape=2x2 or ())

            # case 2: agent did reach its goal
            # first position in curr task (here can have 1 or 2 positions, pickup and/or delivery)
            elif np.array_equal(agent_pos, curr_task[0]):
                curr_task = np.delete(curr_task, 0, 0)  # remove first element
                # 2a: the goal was delivery position -> search for a new task
                if curr_task.size == 0:
                    curr_task = self.assign_closest_task(agent_pos=agent_pos)  # find task (ndarray, shape=2x2 or ())
                # 2b: the goal was pickup position -> delivery pos remaining
                # no need to do anything here

            # case 3) -> not reached its goal yet -> do nothing

            # update the register with curr task
            self.task_register[i] = curr_task.copy()

    def assign_closest_task(self,
                            agent_pos: np.array
                            ) -> np.array:
        """
        Find and assign the available task closest to the agent position
        If a task is found, remove it from active list
        :param agent_pos: shape = (2), agent position in the map
        :return: task with shape = (2, 2), if assigned task
                 array of None, shape = (), if no tasks are available
        """
        # if at least one task is available
        if self.active_task_list:
            # list -> [h-value1, h-value2, ...]
            # h-value from current agent position to pickup_pos of the task
            # argmin -> index of avail_task_list where task has min(h-value)
            min_idx = np.argmin([self.h_coll[tuple(pickup)][tuple(agent_pos)]
                                 for pickup, _ in self.active_task_list
                                 ])
            # pop found task from active list
            task = self.active_task_list.pop(min_idx)

            return task
        else:
            return np.array(())

    def get_goal_list(self) -> np.array:
        """
        Get a list of all agents goals
        A goal is either taken from an assigned task (pickup pos or delivery pos) or from current position,
        which correspond to standing still
        :return: goal list, shape = (agent_num, 2)
                 list in array form for agent_state compatibility
        """
        goal_list = []

        # loop over all agents
        for i in range(self.agent_num):
            task_pos = self.task_register[i]  # array of pos in task register
            # not null
            if task_pos.size:
                goal_list.append(task_pos[0])
            # no task available, set current position
            else:
                goal_list.append(self.curr_agent_pos[i])

        return np.array(goal_list, dtype=np.int8)

    def move_agents(self,
                    action_idx_predict: np.array
                    ) -> None:
        """
        Get index of action for each agent
        Move the agent, if move is allowed
        If the move is not allowed, change the suggested move to a 'wait'
        Update agent schedule and current agent positions
        :param action_idx_predict: shape = (B*N)
                                   list of action index predicted by the model
        """
        # get predicted actions, agent_num x 2
        pred_moves = np.zeros((self.agent_num, 2), dtype=np.int8)

        pred_moves[action_idx_predict == self.up_idx] = self.up
        pred_moves[action_idx_predict == self.down_idx] = self.down
        pred_moves[action_idx_predict == self.left_idx] = self.left
        pred_moves[action_idx_predict == self.right_idx] = self.right
        pred_moves[action_idx_predict == self.stop_idx] = self.stop

        # get new agents predicted positions
        pred_agent_pos = self.curr_agent_pos + pred_moves

        # out of bound pos -> substituted with current agent pos (= stop move)
        pred_agent_pos = np.where((pred_agent_pos < 9) & (pred_agent_pos > 0),  # condition = pos is inbound
                                  pred_agent_pos,  # if condition holds
                                  self.curr_agent_pos)  # if it doesn't hold

        # predicted pos coincide with an obstacle -> substituted with current agent pos (= stop move)
        obstacle_idx = np.where(self.map[pred_agent_pos[:, 0], pred_agent_pos[:, 1]] == 1)
        pred_agent_pos[obstacle_idx] = self.curr_agent_pos[obstacle_idx]

        # move agents
        self.curr_agent_pos = pred_agent_pos.copy()

        # update agent_schedule with current pos and timestep
        for i in range(self.agent_num):
            self.agent_schedule[i].append((self.curr_agent_pos[i, 0],
                                           self.curr_agent_pos[i, 1],
                                           self.timestep))  # timestep

    @staticmethod
    def exp_multinorm(action_vector: torch.Tensor) -> np.array:
        """
        Get action index prediction
        Use action vector as weights for exponential multinomial distribution
        Action key is sampled from the distribution
        :param action_vector: vector with actions probs, shape = (Batch_size*Agent_num, 5)
        :return: array of selected action indexes, shape = (B*N)
        """
        # get exponential
        exp_action_vector = torch.exp(action_vector)
        # 1 action sampled for each B*N
        action_idx_predict = torch.multinomial(exp_action_vector, 1)
        # shape = B*N x 1 -> B*N
        return action_idx_predict.detach().cpu().numpy().squeeze()
