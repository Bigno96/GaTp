"""
Utility functions for experts algorithms
"""

from operator import sub

import numpy as np

# delta dictionary
DELTA = [(-1, 0),  # go up
         (0, -1),  # go left
         (1, 0),  # go down
         (0, 1)]  # go right

# move dictionary
MOVE_LIST = [(-1, 0),  # go up
             (0, -1),  # go left
             (1, 0),  # go down
             (0, 1),  # go right
             (0, 0)]  # stay still


def compute_manhattan_heuristic(input_map, goal):
    """
    Create a matrix the same shape of the input map
    Calculate the cost from the goal node to every other node on the map using MANHATTAN heuristic
    Return the heuristic matrix
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param goal: (x, y), tuple of int with goal cartesian coordinates
    :return: heuristic, np.ndarray, heuristic.shape = input_map.shape
    """
    # abs(current_cell.x – goal.x) + abs(current_cell.y – goal.y)
    heuristic = [(abs(row - goal[0]) + abs(col - goal[1]))
                 for row in range(input_map.shape[0])
                 for col in range(input_map.shape[1])
                 ]
    return np.array(heuristic, dtype=int).reshape(input_map.shape)


def is_valid_expansion(child_pos, input_map, closed_list):
    """
    Check if is possible for A* to expand the new cell
    1) Check if the cell is inside map boundaries
    2) Check if the cell has already been expanded
    3) Check if the cell has an obstacle inside
    :param child_pos: (x, y), int tuple of new position
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param closed_list: implemented as matrix with shape = input_map.shape, np.ndarray
    :return: True if all 3 checks are passed, False else
    """
    x, y = child_pos

    # check 1), x < shape_x & x >= 0 & y >= 0 & y < shape_y
    if x < 0 or x >= input_map.shape[0] or y < 0 or y >= input_map.shape[1]:
        return False

    # check 2), the current node has not been expanded
    if closed_list[child_pos] == 1:
        return False

    # check 3), the current cell is not an obstacle (not 1)
    if input_map[child_pos] == 1:
        return False

    return True


def check_token_conflicts(token, new_pos, curr_pos, new_timestep):
    """
    Check that at new_pos there are no conflicts in token
    :param token: summary of other agents planned paths
                  dict -> {agent_id : path}
                  with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...])
                  x, y -> cartesian coords, t -> timestep
    :param curr_pos: (x, y), int tuple of curr position
    :param new_pos: (x, y), int tuple of new position
    :param new_timestep: int, global timestep of new expansion, always > 0
    :return: True if no conflicts are found, False else
    """
    # if something is not specified, defaults to True
    if not token or not new_pos or not curr_pos or new_timestep is None:
        return True

    # when called, all paths in the token are from a different agent
    # construct list of bad moves
    bad_moves_list = {(x_s, y_s)
                      for path in token.values()
                      for x_s, y_s, t_s in path
                      # no swap constraint
                      if (t_s == (new_timestep-1)  # avoid going into their current position next move
                          # if that agent is going into my current position
                          and curr_pos in [(x_p, y_p)
                                           for x_p, y_p, t_p in path
                                           if t_p == new_timestep]
                          )
                      # avoid node collision
                      or t_s == new_timestep
                      }

    # if attempted move not conflicting, return True
    return new_pos not in bad_moves_list


def preprocess_heuristics(input_map, task_list, non_task_ep_list):
    """
    Since cost-minimal paths need to be found only to endpoints, the path costs from all locations to all endpoints
    are computed in a preprocessing phase
    Manhattan Heuristic is used to estimate path costs
    :param input_map: np.ndarray, type=int, size:H*W, matrix of 0 and 1
    :param task_list: list of tasks -> [(task1), (task2), ...]
                      task: tuple ((x_p,y_p),(x_d,y_d)) -> ((pickup),(delivery))
    :param non_task_ep_list: list of endpoints not belonging to a task -> [(ep1), (ep2), ...]
                             endpoint: tuple (x,y) of int coordinates
    :return: heuristic_collection
             dict -> {endpoint : h_map}
             endpoint: tuple (x,y) of int coordinates
             h_map: np.ndarray, type=int, shape=input_map.shape, heuristic matrices with goal = endpoint
    """
    # task related endpoints
    ep_list = [ep
               for task in task_list
               for ep in task]
    # add non task related endpoints
    ep_list.extend(non_task_ep_list)

    # compute h_map list
    h_map_list = [compute_manhattan_heuristic(input_map=input_map, goal=ep)
                  for ep in ep_list]

    # return dictionary
    return dict(zip(iter(ep_list), iter(h_map_list)))


def free_cell_heuristic(target, input_map, token, target_timestep):
    """
    Get how many cells are free around target at specified timestep
    :param target: int tuple (x,y)
    :param input_map: np.ndarray, matrix of 0s and 1s, 0 -> free cell, 1 -> obstacles
    :param token: summary of other agents planned paths
                      dict -> {agent_id : path}
                      with path = deque([(x_0, y_0, t_0), (x_1, y_1, t_1), ...])
                      x, y -> cartesian coords, t -> timestep
    :param target_timestep: global timestep of the execution
    :return: int, value of free cells around
    """
    # get agent adjacent cells
    target_neighbours = [(target[0]+move[0], target[1]+move[1])
                         for move in DELTA]
    # get token positions at target_timestep
    # agent who called this must not be in the token
    token_pos_list = [(x, y)
                      for path in token.values()
                      for x, y, t in path
                      if t == target_timestep]
    # count and return free cells
    return sum([1
                for pos in target_neighbours
                if (0 <= pos[0] < input_map.shape[0] and 0 <= pos[1] < input_map.shape[1])
                and input_map[pos] == 0
                and pos not in token_pos_list])


def transform_agent_schedule(agent_schedule):
    """
    A matrix-form notation is used to represent produced agent schedule
    This is done in order to feed the neural network of the GaTp agent
    :param agent_schedule: {agent_id : schedule}
                            with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
    :return: matrix -> 5 (5 actions, z-axis) x num_agent (x-axis) x makespan (max path length, y-axis)
    """
    num_agent = len(agent_schedule)
    # get makespan (all paths are the same length, since everyone waits standing still the ending)
    makespan = len(agent_schedule[0])-1

    # matrix -> actions (z-axis) x num_agent (x-axis) x makespan (y-axis)
    # 5 actions order: go_up, go_left, go_down, go_right, stay_still
    matrix = np.zeros(shape=(5, num_agent, makespan), dtype=np.int8)

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
        #   agent -> x coord in np.array, agent number
        #   t -> y coord in np.array, timestep
        #   move_idx -> z coord in np.array, move performed by agent at timestep t
        for t, move_idx in enumerate(move_idx_list):
            matrix[(move_idx, agent, t)] = 1

    return matrix
