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

    # get a new view of agent schedule where all agents' steps are paired by timestep
    # time_view = [ (s1_0, s2_0, s3_0), (s1_1, s2_1, s3_1), ... ]
    # list of tuples of tuples
    time_view = list(zip(*agent_schedule.values()))

    # loop over each system timestep
    # first, check node conflicts
    for time_slice in time_view:
        counter = Counter(time_slice)
        # take only steps that appear more than once (meaning  collision)
        coll_list = [val-1 for val in counter.values()]
        if sum(coll_list) > 0:
            coll_count += sum(coll_list)

    # second, check swap conflicts
    for ag, path in agent_schedule.items():
        for x, y, t in path:
            if t > 0:
                # get (curr_pos, old_pos) for all the other agents, with curr_timestep == t
                other_agent_step = [((x_, y_), (path_[t_-1][0], path_[t_-1][1]))
                                    for ag_, path_ in agent_schedule.items()
                                    for x_, y_, t_ in path_
                                    if ag_ != ag
                                    and t_ == t and t_ > 0
                                    ]
                # check if (old_pos, curr_pos) of the agent is in other_agent_step
                # since it's the inverse of the order in other_agent_step -> checking swap
                if ((path[t-1][0], path[t-1][1]), (x, y)) in other_agent_step:
                    coll_count += 1

    return coll_count
