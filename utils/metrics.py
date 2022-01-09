"""
FIle for computing metrics and evaluating quality of solutions
"""

from collections import Counter


def count_collision(agent_schedule):
    """
    Get all agent's path for a MAPD instance solution and count collisions
    Collision is caused by either node or swap constraint violations
    :param agent_schedule: {agent_id : schedule}
                            with schedule = deque([(x_0, y_0, 0), (x_1, y_1, t_1), ...])
    :return: int, number of collision detected
    """
    coll_count = 0
    time_list_idx = []

    # get a new view of agent schedule where all agents' steps are paired by timestep
    # time_view = [ (a1_0, a2_0, a3_0), (a1_1, a2_1, a3_1), ... ]
    # list of tuples of tuples
    time_view = list(zip(*agent_schedule.values()))

    # first, check node conflicts
    # loop over each system timestep
    for idx, time_slice in enumerate(time_view):
        # cutoff timesteps
        time_slice = [step[:-1] for step in time_slice]
        counter = Counter(time_slice)
        # take only steps that appear more than once (meaning  collision)
        coll_list = [val-1 for val in counter.values()]
        if sum(coll_list) > 0:
            time_list_idx.append(idx)
            coll_count += sum(coll_list)

    return time_list_idx, coll_count


