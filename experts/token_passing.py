"""
Token Passing algorithm

Number of agents cooperating is defined as a function parameter
Path1, Path2 functions in the pseudocode are, respectively:
    - experts.a_star
    - experts.utils.find_resting_pos

Agents, once finished their task, if not assigned a new one, moves to a non-conflicting position
That is, an endpoint such that the delivery locations of all tasks are different from the chosen endpoint,
no path of other agents in the token ends in the chosen endpoint
and does not collide with the paths of other agents stored in the token

The following implementation is based on:
    - Token Passing pseudocode as described in
        Ma, H., Li, J., Kumar, T. K., & Koenig, S. (2017).
        Lifelong multiagent path finding for online pickup and delivery tasks.
        arXiv preprint arXiv:1705.10868.
"""

