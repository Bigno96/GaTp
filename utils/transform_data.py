"""
Utilities for transforming environment data and expert solutions into neural network compatible data

Train data:
    1- Input tensor -> torch.IntTensor,
                       shape = (makespan, num_agent, num_input_channels, FOV+2*border, FOV+2*border)
                       See GaTp/data_loading/agent_state.py for more information about input tensor composition
    2- GSO -> np.ndarray,
              shape = (makespan, num_agent, num_agent)
              Adjacency matrix at each timestep
    3- Target -> np.ndarray,
                 shape = (makespan, num_agent, 5)
                 Matrix representation of agent schedule
                 5 actions: up, down, left, right, wait
                 sequence of moves that describes the policy to learn

Test data:
    1- Start_pos_list -> torch.IntTensor,
                         shape = (agent_num, 2)
                         Agents starting positions
    2- Task_list -> torch.IntTensor,
                    shape = (task_num, 2, 2)
                    Task list, each task has 2 tuple of coordinates, (pickup, delivery)
    3- Makespan -> int
                   Length of the expert solution
    4- Service_time -> float
                       Average timesteps needed for an agent to complete a task
"""

import os
import pickle
import numpy as np

from operator import sub
from utils.agent_state import AgentState
from utils.expert_utils import MOVE_LIST
from utils.graph_utils import compute_adj_matrix


class DataTransformer:
    """
    Read environment data and expert solution from pickle dataset file
    Transform parsed data to prepare for ML Model input
    """

    def __init__(self, config, data_path, mode):
        """
        :param config: configuration Namespace
        :param data_path: path to data folder
        :param mode: str, options: ['test', 'train', 'valid']
        """
        self.config = config
        self.agent_state = AgentState(config=config)
        self.data_path = data_path

        # expert used for solving scenarios
        self.expert_type = config.expert_type

        assert mode in ['test', 'train', 'valid']
        # mode of usage
        self.mode = mode

        # data retrieving function
        if self.mode == 'train':
            self.get_data = self.get_train_data
        else:
            self.get_data = self.get_test_data

    def get_train_data(self, basename):
        """
        Return tuple with all the data necessary for neural network training, appropriately transformed
        train data = (Input tensor, GSO, Target)
        :param basename: str, 'mapID_caseID'
        :return: (Input tensor, GSO, Target)
                 Input tensor -> torch.IntTensor,
                    shape = (makespan, num_agent, num_input_channels, FOV+2*border, FOV+2*border)
                 GSO -> np.ndarray,
                    shape = (makespan, num_agent, num_agent)
                 Target -> np.ndarray,
                    shape = (makespan, num_agent, 5)
        """
        environment, expert_sol = self.load_from_pickle(basename=basename)

        # 1) input tensor
        input_tensor = self.build_train_input_tensor(input_map=environment['map'],
                                                     agent_schedule=expert_sol['agent_schedule'],
                                                     goal_schedule=expert_sol['goal_schedule'])

        # 2) GSO
        GSO = self.compute_gso(agent_schedule=expert_sol['agent_schedule'])

        # 3) target
        target = self.transform_agent_schedule(agent_schedule=expert_sol['agent_schedule'])

        return input_tensor, GSO, target

    def get_test_data(self, basename):
        """
        Return tuple with all the data necessary for neural network testing/validation, appropriately transformed
        test data = (Start_pos_list, Task_list, Makespan, Service_time)
        :param basename: str, 'mapID_caseID'
        :return: (Start_pos_list, Task_list, Makespan, Service_time)
                  Start_pos_list -> torch.IntTensor,
                                    shape = (agent_num, 2)
                  Task_list -> torch.IntTensor,
                               shape = (task_num, 2, 2)
                  Makespan -> int
                  Service_time -> float
        """
        environment, expert_sol = self.load_from_pickle(basename=basename)

        # 1) start_pos_list
        start_pos_list = np.array(environment['start_pos_list'], dtype=np.int8)

        # 2) task_list
        task_list = np.array(environment['task_list'], dtype=np.int8)

        # 3) makespan
        makespan = expert_sol['makespan']

        # 4) service_time
        service_time = expert_sol['service_time']

        return start_pos_list, task_list, makespan, service_time

    def load_from_pickle(self, basename):
        """
        Return environment dict, expert solution dict loaded from pickled file
        :param basename: str, 'mapID_caseID'
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

    @staticmethod
    def schedule_to_numpy(schedule):
        """
        Transform given schedule from dict to numpy array
        :param schedule: dict -> {agent_id : schedule}
                                 with schedule = deque([(pos, 0), (pos, 1), ...])
        :return: np.ndarray, shape = (makespan, num_agents, 2)
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

    def build_train_input_tensor(self, input_map, agent_schedule, goal_schedule):
        """
        Build input tensor for training input data
        Check data_loading/agent_state -> get_sequence_input_tensor for more information
        :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
        :param agent_schedule: dict -> {agent_id : schedule}
                                 with schedule = deque([(x_0, y_0, 0), (x_1, y_1, 1), ...])
        :param goal_schedule: dict -> {agent_id : schedule}
                                with schedule = deque([(current_goal, 0), (curr_goal, 1), ...])
                                curr_goal -> position the agent is trying to reach
        :return: torch Int Tensor of the input configuration
                 input state = makespan x num_agents x input state of agent
        """
        # set obstacle map in agent state
        self.agent_state.set_obstacle_map(input_map=input_map)

        # prepare agent schedule, makespan x num_agents x 2
        agent_pos_schedule = self.schedule_to_numpy(schedule=agent_schedule)

        # prepare goal schedule, makespan x num_agents x 2
        goal_pos_schedule = self.schedule_to_numpy(schedule=goal_schedule)

        return self.agent_state.get_sequence_input_tensor(goal_pos_schedule=goal_pos_schedule,
                                                          agent_pos_schedule=agent_pos_schedule,
                                                          makespan=len(agent_schedule[0]))

    def compute_gso(self, agent_schedule):
        """
        Compute GSO for each timestep
        GSO -> degree normalized adjacency matrix
        :param agent_schedule: {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
        :return: GSO: np.ndarray,
                      shape = (makespan, num_agent, num_agent)
        """
        # init, shape = (makespan, num_agent, num_agent)
        makespan = len(agent_schedule[0])
        agent_num = len(agent_schedule)
        gso = np.zeros(shape=(makespan, agent_num, agent_num))

        # prepare agent schedule, ndarray with shape -> makespan x num_agents x 2
        agent_pos_schedule = self.schedule_to_numpy(schedule=agent_schedule)

        # compute gso for each timestep
        for t in range(makespan):
            gso[t] = compute_adj_matrix(agent_pos_list=agent_pos_schedule[t],
                                        comm_radius=self.config.comm_radius)

        return gso

    @staticmethod
    def transform_agent_schedule(agent_schedule):
        """
        A matrix-form notation is used to represent produced agent schedule
        This is done in order to feed the neural network of the GaTp agent
        :param agent_schedule: {agent_id : schedule}
                                with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
        :return: np.ndarray, shape = (makespan, num_agent, 5)
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
            diff: tuple[int, int]
            move_idx_list = [MOVE_LIST.index(diff)
                             for diff in diff_list]

            # update matrix: actions x num_agent x makespan
            #   agent -> y coord in np.array, agent number
            #   t -> x coord in np.array, timestep
            #   move_idx -> z coord in np.array, move performed by agent at timestep t
            for t, move_idx in enumerate(move_idx_list):
                matrix[(t+1, agent, move_idx)] = 1

        return matrix