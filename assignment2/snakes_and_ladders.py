import sys
sys.path.append("../rl/")

from markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from distribution import Choose

# States are int values strictly in set {0, 1, ..., 100}
# Single terminal state of 100

def create_snake_and_ladders_mp():
    jump_state_mappings = {
        1:38, 4:14, 9:31, 16:6, 21:42, 28:84, 36:44, 47:26, 49:11, 51:67, 56:53, 
        62:19, 64:60, 71:91, 80:100, 87:24, 93:73, 95:75, 98:78
    }
    # these states map instantly to other states due to ladder or snake
    jump_state_lambda = lambda x:  jump_state_mappings[x] if x in jump_state_mappings else x

    transition_map = {100: None}
    for i in range(0, 100, 1):
        next_states = []
        for j in range(1, 7, 1):
            if (i+j) < 101:
                next_states.append(i+j)
        transition_map[i] = Choose(next_states)    

    snakes_and_ladders_mp = FiniteMarkovProcess(transition_map=transition_map)
    return snakes_and_ladders_mp

def traces(snakes_and_ladders_mp):
    start_state_distribution = Choose([0])
    traces = snakes_and_ladders_mp.traces(start_state_distribution=start_state_distribution)
    return traces


def create_snake_and_ladders_mrp():
    jump_state_mappings = {
        1:38, 4:14, 9:31, 16:6, 21:42, 28:84, 36:44, 47:26, 49:11, 51:67, 56:53, 
        62:19, 64:60, 71:91, 80:100, 87:24, 93:73, 95:75, 98:78
    }
    # these states map instantly to other states due to ladder or snake
    jump_state_lambda = lambda x:  jump_state_mappings[x] if x in jump_state_mappings else x

    transition_map = {100: None}
    for i in range(0, 100, 1):
        next_states = []
        for j in range(1, 7, 1):
            if (i+j) < 101:
                next_states.append(i+j)
        transition_map[i] = Choose([(x, 1) for x in next_states])    

    snakes_and_ladders_mrp = FiniteMarkovRewardProcess(transition_reward_map=transition_map)
    return snakes_and_ladders_mrp