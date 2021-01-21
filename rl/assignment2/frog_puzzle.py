import sys
sys.path.append("..")

from markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from distribution import Choose


def create_frog_puzzle(n=10):
    transition_map = {0: None}
    for i in range(1, n+1, 1):
        transition_map[i] = Choose(list(range(0, i, 1)))    

    frog_puzzle_mp = FiniteMarkovProcess(transition_map=transition_map)
    return frog_puzzle_mp


def traces(frog_puzzle_mp, n):
    start_state_distribution = Choose([n])
    traces = frog_puzzle_mp.traces(start_state_distribution=start_state_distribution)
    return traces