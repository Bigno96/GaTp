"""
Class for handling Multi Agent Execution Simulation

Supported ops:
    - Keep track of all agents position
    - Handle task assignment
    - Move agents following learned Action Policy
"""

import torch
import numpy as np

from itertools import repeat
from collections import deque

from utils.agent_state import AgentState
from utils.graph_utils import compute_adj_matrix
from utils.expert_utils import preprocess_heuristics


class MultiAgentSimulator:
    """
    Supporting operations for simulation
    Used to simulate a solution of a MAPD problem during validation/testing

    Agents will move accordingly to learned action policy
    Task assignment is centralized
    """

    def __init__(self, config):
        """
        :param config: configuration Namespace
        """
        self.config = config

        '''basic'''
        self.agent_num = self.config.agent_number
        # NN trained model
        self.model = None
        # agent state representation
        self.agent_state = AgentState(config=self.config)
        # obstacle map
        self.map = None  # ndarray, H x W

        '''simulation data structures'''
        self.agent_start_pos = None  # ndarray, agent_num x 2
        self.curr_agent_pos = None  # ndarray, agent_num x 2

        self.task_register = {}  # dictionary for keeping track of task assignment, id : ndarray (2x2 or 1x2 or ())
        self.task_list = None  # ndarray, task_num x 2 x 2
        self.active_task_list = []  # list of arrays for tracking active tasks
        self.new_task_pool = deque()  # deque of arrays for new tasks

        self.agent_schedule = {}  # dictionary for keeping track of agent movements

        self.h_coll = {}  # dictionary of precomputed distance heuristics

        '''simulation parameters'''
        self.timestep = 0  # timestep counter
        self.terminate = False  # terminate boolean

        self.task_number = self.config.task_number  # total number of tasks
        self.activated_task_count = 0  # number of tasks activated

        self.split_idx = int(self.config.imm_task_split * self.task_number)  # % of task to immediately activate
        self.new_task_per_timestep = self.config.new_task_per_timestep  # how many tasks to add at each timestep
        self.step_between_insertion = self.config.step_between_insertion  # how many timestep between each insertion

        '''pre-define directions'''
        self.up = np.array([-1, 0], dtype=np.int8)  # int arrays
        self.down = np.array([1, 0], dtype=np.int8)
        self.left = np.array([0, -1], dtype=np.int8)
        self.right = np.array([0, 1], dtype=np.int8)
        self.stop = np.array([0, 0], dtype=np.int8)
        self.up_idx = 0  # indexes for action selection out of model prediction
        self.left_idx = 1
        self.down_idx = 2
        self.right_idx = 3
        self.stop_idx = 4

    def set_up_simulation(self, obstacle_map, ag_start_pos, task_list, model):
        """
        Set up variables for the simulation
        :param obstacle_map: IntTensor, shape = (H, W)
        :param ag_start_pos: IntTensor, shape = (agent_num, 2)
        :param task_list: IntTensor, shape = (task_num, 2, 2)
        :param model: torch.nn.Module, trained model
        """
        # fix model for evaluation mode
        self.model = model
        self.model.eval()

        # init map and set it for agent state
        self.map = obstacle_map.detach().cpu().numpy()  # convert it to numpy
        self.agent_state.set_obstacle_map(input_map=self.map)

        # scalar variables init
        self.terminate = False
        self.timestep = 0  # time 0 is for setup
        self.activated_task_count = self.split_idx  # update activated count with immediate tasks

        # agent start and current position
        self.agent_start_pos = ag_start_pos.detach().cpu().numpy()  # convert it to numpy
        self.curr_agent_pos = self.agent_start_pos.copy()  # copy to avoid memory interference

        # agent schedule init with agent start pos
        for i in range(self.agent_num):
            self.agent_schedule[i] = [(self.agent_start_pos[i, 0],
                                       self.agent_start_pos[i, 1],
                                       0)]  # timestep
        # init goal register as empty
        self.task_register = dict(zip(range(self.agent_num), repeat(np.array(()))))

        # task init
        self.task_list = task_list.detach().cpu().numpy()  # convert it to numpy
        self.active_task_list = list(self.task_list[:self.split_idx])  # activate first n tasks
        self.new_task_pool = deque(self.task_list[self.split_idx:])  # put the rest in the waiting pool

        # precompute distance heuristics towards all possible endpoints
        self.h_coll = preprocess_heuristics(input_map=self.map,
                                            task_list=self.task_list,
                                            non_task_ep_list=[])  # agents don't use parking positions

    def execute_one_timestep(self):
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
        # intTensor -> (num_agents, 3 (channels), FOV+2*border, FOV+2*border)
        input_tensor = self.agent_state.get_input_tensor(goal_pos_list=goal_list,
                                                         agent_pos_list=self.curr_agent_pos)
        input_tensor = input_tensor.unsqueeze().to(self.config.device)  # shape = 1 x N x 3 x F_H x F_W

        # obtain and set gso
        GSO = compute_adj_matrix(agent_pos_list=self.curr_agent_pos,
                                 comm_radius=self.config.comm_radius)
        GSO = torch.from_numpy(GSO).to(self.config.device)
        self.model.set_gso(GSO)

        # predict with model
        model_output = self.model(input_tensor)  # B*N x 5 -> since B=1, N x 5

        # exp_multinorm for getting predicted action
        action_idx_predict = self.exp_multinorm(model_output)  # 1*N x 1 (since B=1)

        # move agents
        self.move_agents(action_idx_predict)

        # check end condition
        if (self.activated_task_count == self.task_number   # all tasks have been added
                and not self.active_task_list   # all tasks have been assigned
                and not any(arr.size for arr in self.task_register.values())):  # all agents have finished their task
            self.terminate = True

    def update_task_register(self):
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

    def assign_closest_task(self, agent_pos):
        """
        Find and assign the available task closest to the agent position
        If a task is found, remove it from active list
        :param agent_pos: np.ndarray, shape = (2), agent position in the map
        :return: np.ndarray, shape = (2, 2), if assigned task
                 np.ndarray of None, shape = (), if no tasks are available
        """
        # if at least one task is available
        if self.active_task_list:
            # list -> [h-value1, h-value2, ...]
            # h-value from current agent position to pickup_pos of the task
            # argmin -> index of avail_task_list where task has min(h-value)
            task = self.active_task_list[np.argmin([self.h_coll[tuple(pickup)][agent_pos]  # tuple for hash
                                                    for pickup, _ in self.active_task_list
                                                    ])]
            # remove found task from active list
            self.active_task_list.remove(task)

            return task
        else:
            return np.array(())

    def get_goal_list(self):
        """
        Get a list of all agents goals
        A goal is either taken from an assigned task (pickup pos or delivery pos) or from current position,
        which correspond to standing still
        :return: np.ndarray, shape = (agent_num, 2)
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

    def move_agents(self, action_idx_predict):
        """
        Get index of action for each agent
        Move the agent, if move is allowed
        If the move is not allowed, change the suggested move to a 'wait'
        Update agent schedule and current agent positions
        :param action_idx_predict: np.ndarray, shape = (B*N)
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
    def exp_multinorm(action_vector):
        """
        Get action index prediction
        Use action vector as weights for exponential multinomial distribution
        Action key is sampled from the distribution
        :param action_vector: FloatTensor, shape = (Batch_size*Agent_num, 5)
        :return: np.ndarray, shape = (B*N)
        """
        # get exponential
        exp_action_vector = torch.exp(action_vector)
        # 1 action sampled for each B*N
        action_idx_predict = torch.multinomial(exp_action_vector, 1)
        # shape = B*N x 1 -> B*N
        return action_idx_predict.detach().cpu().numpy().squeeze()
