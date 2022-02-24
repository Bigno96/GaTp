"""
Class for Agent State representation utilities

4 different maps configurations:
- original map -> shape = (H, W)
- padded map -> shape = (H + FOV/2, W + FOV/2), original map but padded with half the FOV around
- FOV map -> shape = (FOV, FOV), cut from padded map around agent position
- padded FOV map -> shape = (FOV + 2*border, FOV + 2*border), FOV map padded with a border around

Input tensor to feed to each agent -> 3 channels
Each channel -> padded FOV map with different information
Respectively:

    1- obstacle map: 1s for obstacle, 0 else.
                     The outside of the original map is padded with 1s, obstacles
                     Border is filled with 0s, not used by this channel but dimensions have to be the same

    2- goal map: 1 for the agent's goal, 0 else.
                 Always one and only one goal
                 The outside of the original map is padded with 0s, no goal outside
                 Border is used when the goal is outside FOV map to project it
                 If agent has no proper goal, its goal is its current position, meaning it will stand still

    3- agents map: 1s for agents' positions, 0 else.
                   Keep track of his relative position and relative position of the other agents
                   The outside of the original map is padded with 0s, no agents outside
                   Border is filled with 0s, not used by this channel but dimensions have to be the same
"""

import numpy as np


class AgentState:
    """
    Allow representing agent's state
    Keep tracks of obstacle map and 'local' objective (in the FOV) of each agent
    Allow retrieving input tensor to feed to an agent
    """

    def __init__(self, config):
        self.config = config

        self.agent_number = self.config.agent_number
        self.FOV = self.config.FOV                  # Field Of View radius
        self.FOV_width = int(self.FOV/2)            # half radius, used for padding
        self.border = 1                             # size of the border outside FOV, to project goal
        self.H = self.FOV + int(2*self.border)      # height of agents available map
        self.W = self.FOV + int(2*self.border)      # width of agents available map
        self.dist = int(np.floor(self.W/2))         # distance of agents from the outside of the available map
        self.center_x = self.dist                   # map center position, coinciding with agent relative pos
        self.center_y = self.dist

        self.obstacle_map = None               # map of the obstacles, shape=(H, W)
        self.obstacle_map_pad = None        # padded obstacles map, shape=(H+FOV_width, W+FOV_width)

        # objectives of all the agents within their FOV (either goal projection or curr pos)
        # shape = (num_agents, 2)
        self.local_objective_list = None         # local objectives with global coordinates

    @staticmethod
    def simple_pad(input_array, pad_width, fill_value):
        """
        Fast version of numpy pad, padding with constant value around
        Code taken from np.pad implementation
        Avoid all the np.pad memory copies, necessary for other functionalities, not for constant padding
        Speed up for larger databases
        Return a copy of the original array, padded
        :param input_array: np.ndarray, array to pad
        :param pad_width: int, value to add to each axis
        :param fill_value: int, value to pad with
        :return: np.ndarray, shape = input_array.shape + pad_width (over all axis)
        """
        # allocate grown array
        new_shape = tuple(pad_width + s + pad_width
                          for s in input_array.shape)
        padded = np.empty(new_shape, dtype=input_array.dtype)

        padded.fill(fill_value)

        # copy old array into correct space
        original_area_slice = tuple(slice(pad_width, pad_width + s)
                                    for s in input_array.shape)
        padded[original_area_slice] = input_array

        return padded

    def set_obstacle_map(self, input_map):
        """
        Set obstacle maps (padded and not) in the current agent state
        Padding is added outside the map, with value = 1, to simulate obstacles and force the agent to stay inside
        :param input_map: np.ndarray, shape=(H, W)
        """
        # map of all the obstacles for the agent
        self.obstacle_map = input_map.copy()
        # pad obstacle map with half the FOV all around
        # pad value = 1 -> all obstacles outside
        self.obstacle_map_pad = self.simple_pad(input_array=input_map, pad_width=self.FOV_width,
                                                fill_value=1).astype(np.int8)

    def get_agent_pos_map(self, agent_pos_list):
        """
        Compute and return a map with marked ONLY agents position
        Map is padded in the outside
        Obstacle map is assumed to be set
        :param agent_pos_list: list of agents current position, np.ndarray, shape=(num_agents, 2)
        :return: np.ndarray, shape=(H+FOV_width, W+FOV_width), 1s in each agent position, 0s else
        """
        # verify that obstacle map is assigned
        assert self.obstacle_map is not None
        # agent_pos_map = np.zeros([self.W, self.H], dtype=np.int8)
        agent_pos_map = np.zeros_like(self.obstacle_map, dtype=np.int8)

        # for each agent, read its position and write 1 in the map
        for id_agent in range(self.agent_number):
            pos_x = agent_pos_list[id_agent][0]
            pos_y = agent_pos_list[id_agent][1]
            agent_pos_map[(pos_x, pos_y)] = 1

        # pad with 0, since 1 means that an agent is in that position -> no agents outside
        agent_pos_map_pad = self.simple_pad(input_array=agent_pos_map, pad_width=self.FOV_width, fill_value=0)

        return agent_pos_map_pad

    def project_goal(self, goal_map_FOV, agent_pos, goal_pos):
        """
        Add goal projection to the goal FOV map of the agent, centered around its current position
        First, pad with border, than look for global goal position and add its projection to the border
        Assume goal_pos is outside the FOV
        :param goal_map_FOV: map with all agents' goals
                             trimmed within FOV radius around current agent position
                             np.ndarray, shape=(FOV, FOV)
        :param agent_pos: int tuple, current position of agent
        :param goal_pos: int tuple, position of goal for the agent
                         assumed outside FOV
        :return: padded goal_map_FOV with projected goal
                 np.ndarray, shape=(FOV+border, FOV+border)
        """
        # pad with a border around
        padded_goal_map_FOV = self.simple_pad(input_array=goal_map_FOV, pad_width=self.border, fill_value=0)

        # compute position differentials
        dx = float(goal_pos[0] - agent_pos[0])
        dy = float(goal_pos[1] - agent_pos[1])
        x_sign = np.sign(dx)
        y_sign = np.sign(dy)

        # angle between position of agent and goal
        angle = np.arctan2(dy, dx)

        # if 45° <= angle <= 135° or -135° <= angle <= -45° -> dx >= dy
        if (np.pi/4 <= angle <= np.pi*(3/4)) or (-np.pi*(3/4) <= angle <= -np.pi/4):
            # center + offset -> offset = dist * tan(angle)
            goal_x = int(self.center_x + np.round(self.dist * dx / np.abs(dy)))
            goal_y = int(self.dist * (y_sign + 1))      # either 0 or 2*dist
        # else, dy >= dx
        else:
            # center + offset -> offset = dist * tan(angle)
            goal_y = int(self.center_y + np.round(self.dist * dy / np.abs(dx)))
            goal_x = int(self.dist * (x_sign + 1))      # either 0 or 2*dist

        # set to 1 the projection in the border
        padded_goal_map_FOV[goal_x][goal_y] = 1

        return padded_goal_map_FOV

    def build_input_state(self, load_state):
        """
        Build a padded, FOV version of the 3 input channels (maps): obstacle map, goal map, agents pos map
        Update local objective list with global coordinates of each agent goal
        An agent is always assumed to have a goal!!
        For IDLE agents, goal coincide with current position
        :param load_state: tuple, agent state summary
                           goal positions: np.ndarray, shape=(num_agents, 2)
                           agent positions: np.ndarray, shape=(num_agents, 2)
                           padded map of agent positions: np.ndarray, shape=(H+FOV_width, W+FOV_width)
                           id_agent: int, id of the agent
        :return: input state for the current agent, local objective of the current agent
                 input state: [obstacle_map, goal_map, agent_pos_map]
                 map shape -> (FOV+border, FOV+border)
                 local_objective -> coord of the FOV goal, using original non-padded map coordinates reference
        """
        # unpack agent state
        goal_pos_list, agent_pos_list, agent_pos_map_pad, id_agent = load_state

        # get agent global coordinates, over the non-padded map
        agent_x_global = int(agent_pos_list[id_agent][0])
        agent_y_global = int(agent_pos_list[id_agent][1])

        # get goal global coordinates, over the non-padded map
        goal_x_global = int(goal_pos_list[id_agent][0])
        goal_y_global = int(goal_pos_list[id_agent][1])

        # build FOV coordinates, centered around agent global coords
        # the FOV is all skewed by +FOV_width since it's cut from a padded map
        # since agent_pos_global is based on non-padded coordinate, to have it centered needs to sum FOV_width
        FOV_x = [agent_x_global, agent_x_global + int(2*self.FOV_width) + 1]  # +1 since slicing exclude
        FOV_y = [agent_y_global, agent_y_global + int(2*self.FOV_width) + 1]

        # build obstacle 'channel', cut FOV and pad a border around
        # padded with values=0 since it's not used here but all channels needs to have same size
        obstacle_map_FOV = self.obstacle_map_pad[FOV_x[0]:FOV_x[1], FOV_y[0]:FOV_y[1]]  # cut the FOV out
        obstacle_map_FOV_pad = self.simple_pad(input_array=obstacle_map_FOV, pad_width=self.border, fill_value=0)

        # build agent position 'channel', cut FOV and pad a border around
        # padded with values=0 since it's not used here but all channels needs to have same size
        agent_pos_map_FOV = agent_pos_map_pad[FOV_x[0]:FOV_x[1], FOV_y[0]:FOV_y[1]]  # cut the FOV out
        agent_pos_map_FOV_pad = self.simple_pad(input_array=agent_pos_map_FOV, pad_width=self.border, fill_value=0)

        # build goal 'channel', cut FOV and pad a border around
        goal_pos_map = np.zeros_like(self.obstacle_map, dtype=np.int8)  # full map
        goal_pos_map[goal_x_global][goal_y_global] = 1  # set to 1 the agent goal
        goal_pos_map_pad = self.simple_pad(input_array=goal_pos_map, pad_width=self.FOV_width, fill_value=0)
        goal_pos_map_FOV = goal_pos_map_pad[FOV_x[0]:FOV_x[1], FOV_y[0]:FOV_y[1]]  # cut the FOV out

        # if the goal is in the FOV, just pad it with the border set to 0
        if np.any(goal_pos_map_FOV > 0):
            goal_pos_map_FOV_pad = self.simple_pad(input_array=goal_pos_map_FOV, pad_width=self.border,
                                                   fill_value=0)
        # goal outside the FOV, project it on the border
        else:
            goal_pos_map_FOV_pad = self.project_goal(goal_map_FOV=goal_pos_map_FOV,
                                                     agent_pos=[agent_x_global, agent_y_global],
                                                     goal_pos=[goal_x_global, goal_y_global])

        # get position of the only goal
        # goal_x, goal_y = [x], [y] -> unpack array
        goal_x_FOV, goal_y_FOV = np.nonzero(goal_pos_map_FOV_pad)
        goal_x_FOV, goal_y_FOV = goal_x_FOV[0], goal_y_FOV[0]       # unpack

        # local objective setup, with global coordinates on non-padded coordinates
        # if it's in the fov -> local_obj = (goal_x_global, goal_y_global)
        # if it's outside the fov -> projection -> local_obj = global coord of the projection
        local_objective = np.array((goal_x_FOV + (agent_x_global - self.FOV_width) - 1,
                                    goal_y_FOV + (agent_y_global - self.FOV_width) - 1),
                                   dtype=np.int8)

        # build input state and return it, with local objective (non-padded coords)
        input_state_current_agent = [obstacle_map_FOV_pad, goal_pos_map_FOV_pad, agent_pos_map_FOV_pad]

        return input_state_current_agent, local_objective

    def get_local_obj_list(self):
        """
        :return: shape = (num_agents, 2)
        """
        return self.local_objective_list

    def get_input_tensor(self, goal_pos_list, agent_pos_list):
        """
        Build input state of all the agents in tensor form
        Input state of one agent -> 3 'channels' (maps): obstacle map, agent pos map, goal map
        Update local objective list with the global coordinates of the local objective of each agent
        :param goal_pos_list: list of agents goal position, np.ndarray, shape=(num_agents, 2)
        :param agent_pos_list: list of agents current position, np.ndarray, shape=(num_agents, 2)
        :return: np.ndarray of the input configuration
                 input_tensor.shape = (num_agents, 3 (channels), FOV+2*border, FOV+2*border)
        """
        # get map with agents positions, padded in the outside
        agent_pos_map_pad = self.get_agent_pos_map(agent_pos_list)
        # matrix for holding local paths, num_agents x 2 (coordinates)
        self.local_objective_list = np.zeros([self.agent_number, 2], dtype=np.int8)

        input_state = []
        for id_agent in range(self.agent_number):
            # agent state summary
            load_state = (goal_pos_list, agent_pos_list, agent_pos_map_pad, id_agent)
            # build agent state stacking the 3 maps and get local objective of the agent
            input_state_current_agent, local_objective = self.build_input_state(load_state)
            # collect all the input states
            input_state.append(input_state_current_agent)
            # updates local paths
            self.local_objective_list[id_agent, :] = local_objective

        # transform input state (list of ndarray) into a ndarray,
        # since creating a tensor from a list of ndarray is extremely slow
        input_tensor = np.array(input_state, dtype=np.int8)

        return input_tensor

    def get_sequence_input_tensor(self, goal_pos_schedule, agent_pos_schedule, makespan):
        """
        Build one input state of all the agents in tensor form for each timestep in the schedule
        Input state of one agent -> 3 'channels' (maps): obstacle map, agent pos map, goal map
        :param goal_pos_schedule: schedule (sequence) of agent's goal positions
                                  np.ndarray, shape = (makespan, num_agents, 2)
        :param agent_pos_schedule: schedule (sequence) of agents positions
                                   np.ndarray, shape = (makespan, num_agents, 2)
        :param makespan: length of the agents schedule
        :return: np.ndarray of the input configuration
                 input state = makespan x num_agents x input state of agent
        """
        input_step_list = []

        for t in range(makespan):
            # get list of agent position at current timestep
            agent_pos_list = agent_pos_schedule[t][:]
            # get list of goal position at current timestep
            goal_pos_list = goal_pos_schedule[t][:]
            # get padded map of all the agent positions at current timestep
            agent_pos_map_pad = self.get_agent_pos_map(agent_pos_list)

            input_step = []
            for id_agent in range(self.agent_number):
                # agent state summary
                load_state = (goal_pos_list, agent_pos_list, agent_pos_map_pad, id_agent)
                # build agent state stacking the 3 maps and get local objective of the agent
                input_step_current_agent, _ = self.build_input_state(load_state)
                # collect all the input states for each agent
                input_step.append(input_step_current_agent)

            # collect all the input states for each timestep
            input_step_list.append(input_step)

        # transform input state (list of ndarray) into a ndarray,
        # since creating a tensor from a list of ndarray is extremely slow
        input_tensor = np.array(input_step_list, dtype=np.int8)

        return input_tensor
