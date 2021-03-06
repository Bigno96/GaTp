"""
Utilities for transforming environment data and expert solutions into neural network compatible data

Train data:
    1- Input_state -> np.ndarray,
                      shape = (makespan, num_agent, num_input_channels, FOV+2*border, FOV+2*border)
                      See GaTp/utils/agent_state.py for more information about input tensor composition
    2- GSO -> np.ndarray,
              shape = (makespan, num_agent, num_agent)
              Adjacency matrix at each timestep
    3- Target -> np.ndarray,
                 shape = (makespan, num_agent, 5)
                 Matrix representation of agent schedule
                 5 actions: up, down, left, right, wait
                 sequence of moves that describes the policy to learn

Test data:
    1- Obstacle_map -> np.ndarray,
                       shape = (H, W)
                       Obstacle map, 1s for obstacles, 0s elsewhere
    2- Start_pos_list -> np.ndarray,
                         shape = (agent_num, 2)
                         Agents starting positions
    3- Task_list -> np.ndarray,
                    shape = (task_num, 2, 2)
                    Task list, each task has 2 tuple of coordinates, (pickup, delivery)
    4- Makespan -> int
                   Length of the expert solution
    5- Service_time -> float
                       Average timesteps needed for an agent to complete a task
"""

import os
import pickle

import numpy as np
import utils.agent_state as ag_state
import utils.expert_utils as exp_utils
import utils.graph_utils as g_utils

from operator import sub
from easydict import EasyDict
from typing import Tuple, Dict, Deque


class DataTransformer:
    """
    Read environment data and expert solution from pickle dataset file
    Transform parsed data to prepare for ML Model input
    """

    def __init__(self,
                 config: EasyDict,
                 data_path: str,
                 mode: str):
        """
        :param config: configuration Namespace
        :param data_path: path to data folder
        :param mode: 'test', 'train' or 'valid'
        """
        self.config = config
        self.agent_state = ag_state.AgentState(config=config)
        self.data_path = data_path

        # expert used for solving scenarios
        self.expert_type: str = config.expert_type

        assert mode in ['test', 'train', 'valid']
        # mode of usage
        self.mode = mode

        # data retrieving function
        if self.mode == 'train':
            self.get_data = self.get_train_data
        else:
            self.get_data = self.get_test_data

    def get_train_data(self,
                       basename: str
                       ) -> Tuple[np.array, np.array, np.array]:
        """
        Return tuple with all the data necessary for neural network training, appropriately transformed
        Train data = (Input state, GSO, Target)
        :param basename: 'mapID_caseID'
        :return: (Input state, GSO, Target)
                 Input state -> shape = (makespan, num_agent, num_input_channels, FOV+2*border, FOV+2*border)
                 GSO -> shape = (makespan, num_agent, num_agent)
                 Target -> shape = (makespan, num_agent, 5)
        """
        environment, expert_sol = self.load_from_pickle(basename=basename)

        # 1) input tensor
        input_state = self.build_train_input_state(input_map=environment['map'],
                                                   agent_schedule=expert_sol['agent_schedule'],
                                                   goal_schedule=expert_sol['goal_schedule'])

        # 2) GSO
        GSO = self.compute_gso(agent_schedule=expert_sol['agent_schedule'])

        # 3) target
        target = self.transform_agent_schedule(agent_schedule=expert_sol['agent_schedule'])

        return input_state, GSO, target

    def get_test_data(self,
                      basename: str
                      ) -> Tuple[np.array, np.array, np.array, int, float]:
        """
        Return tuple with all the data necessary for neural network testing/validation, appropriately transformed
        Test data = (Start_pos_list, Task_list, Makespan, Service_time)
        :param basename: str, 'mapID_caseID'
        :return: (Obstacle_map, Start_pos_list, Task_list, Makespan, Service_time)
                  Obstacle_map -> np.ndarray,
                                  shape = (H, W)
                  Start_pos_list -> np.ndarray,
                                    shape = (agent_num, 2)
                  Task_list -> np.ndarray,
                               shape = (task_num, 2, 2)
                  Makespan -> int
                  Service_time -> float
        """
        environment, expert_sol = self.load_from_pickle(basename=basename)

        # 1) obstacle_map
        obstacle_map = environment['map']   # already np.array, dtype=np.int8

        # 2) start_pos_list
        start_pos_list = np.array(environment['start_pos_list'], dtype=np.int8)

        # 3) task_list
        task_list = np.array(environment['task_list'], dtype=np.int8)

        # 4) makespan
        makespan = expert_sol['makespan']

        # 5) service_time
        service_time = expert_sol['service_time']

        return obstacle_map, start_pos_list, task_list, makespan, service_time

    def load_from_pickle(self,
                         basename: str
                         ) -> Tuple[EasyDict, EasyDict]:
        """
        Return environment dict, expert solution dict loaded from pickled file
        :param basename: 'mapID_caseID'
        :return: environment = {
                    'name': file path to the env data file,
                    'map': created map,
                    'start_pos_list': agents starting positions,
                    'parking_spot_list': extra non task related endpoints (excluding agents starting points),
                    'task_list': task list built over the map
                    }
                 expert_data = {
                    'name': file path to the expert data file,
                    'makespan': length of the solution,
                    'service_time': average timesteps required to complete a task,
                    'runtime_per_timestep': ms required to execute a timestep of the expert algorithm,
                    'collisions': number of collision occurred,
                    'agent_schedule': agent action schedule,
                    'goal_schedule': schedule of objectives (goal positions) pursued by agents
                    }
        """
        # get file base name
        file_basename = os.path.join(self.data_path, self.mode, basename)

        # get environment
        with open(file_basename, 'rb') as f:
            environment = pickle.load(f)

        # get expert solution
        expert_sol_path = f'{file_basename}_{self.expert_type}_sol'
        with open(expert_sol_path, 'rb') as f:
            expert_sol = pickle.load(f)

        return environment, expert_sol

    def get_online_train_data(self,
                              input_map: np.array,
                              expert_sol: Dict[str, Dict]
                              ) -> Tuple[np.array, np.array, np.array]:
        """
        Called during online dataset aggregation
        Return tuple with all the data necessary for neural network training, appropriately transformed
        Train data = (Input state, GSO, Target)
        :param input_map: obstacle map
        :param expert_sol: solution of the problem given by an expert algorithm
        :return: (Input state, GSO, Target)
                 Input state -> shape = (makespan, num_agent, num_input_channels, FOV+2*border, FOV+2*border)
                 GSO -> shape = (makespan, num_agent, num_agent)
                 Target -> shape = (makespan, num_agent, 5)
        """
        # 1) input tensor
        input_state = self.build_train_input_state(input_map=input_map,
                                                   agent_schedule=expert_sol['agent_schedule'],
                                                   goal_schedule=expert_sol['goal_schedule'])

        # 2) GSO
        GSO = self.compute_gso(agent_schedule=expert_sol['agent_schedule'])

        # 3) target
        target = self.transform_agent_schedule(agent_schedule=expert_sol['agent_schedule'])

        return input_state, GSO, target

    def build_train_input_state(self,
                                input_map: np.array,
                                agent_schedule: Dict[int, Deque[Tuple[int, int, int]]],
                                goal_schedule: Dict[int, Deque[Tuple[int, int, int]]]
                                ) -> np.array:
        """
        Build input tensor for training input data
        Check data_loading/agent_state -> get_sequence_input_state for more information
        :param input_map: matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
        :param agent_schedule: dict -> {agent_id : schedule}
                                 with schedule = deque([(x_0, y_0, 0), (x_1, y_1, 1), ...])
        :param goal_schedule: dict -> {agent_id : schedule}
                                with schedule = deque([(current_goal, 0), (curr_goal, 1), ...])
                                curr_goal -> position the agent is trying to reach
        :return: input_state shape = makespan x num_agents x input state of agent
        """
        # set obstacle map in agent state
        self.agent_state.set_obstacle_map(input_map=input_map)

        # prepare agent schedule, makespan x num_agents x 2
        agent_pos_schedule = self.schedule_to_numpy(schedule=agent_schedule)

        # prepare goal schedule, makespan x num_agents x 2
        goal_pos_schedule = self.schedule_to_numpy(schedule=goal_schedule)

        return self.agent_state.get_sequence_input_state(goal_pos_schedule=goal_pos_schedule,
                                                         agent_pos_schedule=agent_pos_schedule,
                                                         makespan=len(agent_schedule[0]))

    def compute_gso(self,
                    agent_schedule: Dict[int, Deque[Tuple[int, int, int]]]
                    ) -> np.array:
        """
        Compute GSO for each timestep
        GSO -> degree normalized adjacency matrix
        :param agent_schedule: {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
        :return: GSO, shape = (makespan, num_agent, num_agent)
        """
        # init, shape = (makespan, num_agent, num_agent)
        makespan = len(agent_schedule[0])
        agent_num = len(agent_schedule)
        gso = np.zeros(shape=(makespan, agent_num, agent_num))

        # prepare agent schedule, ndarray with shape -> makespan x num_agents x 2
        agent_pos_schedule = self.schedule_to_numpy(schedule=agent_schedule)

        # compute gso for each timestep
        for t in range(makespan):
            gso[t] = g_utils.compute_adj_matrix(agent_pos_list=agent_pos_schedule[t],
                                                comm_radius=self.config.comm_radius)

        return gso

    @staticmethod
    def transform_agent_schedule(agent_schedule: Dict[int, Deque[Tuple[int, int, int]]]
                                 ) -> np.array:
        """
        A matrix-form notation is used to represent produced agent schedule
        This is done in order to feed the neural network of the GaTp agent
        :param agent_schedule: {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
        :return: matrix form agent schedule,
                 shape = (makespan, num_agent, 5)
        """
        num_agent = len(agent_schedule)
        # get makespan (all paths are the same length, since everyone waits standing still the ending)
        makespan = len(agent_schedule[0])

        # matrix -> makespan x num_agent x actions
        # 5 actions: go_up, go_left, go_down, go_right, stay_still
        matrix = np.zeros(shape=(makespan, num_agent, 5), dtype=np.int8)
        matrix[0, :, 4] = 1     # set first agent action to 'stand still'

        # iterate over all agent's schedules
        for agent, schedule in agent_schedule.items():

            # remove timesteps
            schedule = [(x, y) for (x, y, t) in schedule]
            # this will pair schedule[i] with schedule[i-1], starting from i = 1
            zip_offset = list(zip(schedule[1:], schedule))
            # get difference between each pair in zip_offset
            diff_list = [tuple(map(sub, a, b))
                         for (a, b) in zip_offset]

            # get corresponding index in moves dictionary
            move_idx_list = [exp_utils.MOVE_LIST.index(diff)
                             for diff in diff_list]

            # update matrix: actions x num_agent x makespan
            #   agent -> y coord in np.array, agent number
            #   t -> x coord in np.array, timestep
            #   move_idx -> z coord in np.array, move performed by agent at timestep t
            for t, move_idx in enumerate(move_idx_list):
                matrix[(t+1, agent, move_idx)] = 1

        return matrix

    @staticmethod
    def schedule_to_numpy(schedule: Dict[int, Deque[Tuple[int, int, int]]]
                          ) -> np.array:
        """
        Transform given schedule from dict to numpy array
        :param schedule: dict -> {agent_id : schedule}
                                 with schedule = deque([(x, y, 0), (x, y, 1), ...])
        :return: converted schedule to numpy array,
                 shape = (makespan, num_agents, 2)
        """
        # strip timesteps from agent schedule and convert to ndarray
        pos_schedule = [[step[:-1]  # remove timestep at each step
                         for step in schedule
                         ]
                        for schedule in schedule.values()  # for each agent
                        ]

        pos_schedule = np.array(pos_schedule, dtype=np.int8)
        # reshape: num_agents x makespan x 2 -> makespan x num_agents x 2
        return np.swapaxes(pos_schedule, axis1=0, axis2=1)
