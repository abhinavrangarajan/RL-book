from dataclasses import dataclass
import sys
sys.path.append("../rl/")

from typing import Callable, Optional, Tuple
from markov_process import MarkovProcess, MarkovRewardProcess
from distribution import Categorical, Distribution, Choose
from gen_utils.common_funcs import get_unit_sigmoid_func

# From Ashwin Rao textbook

@dataclass(frozen=True)
class StateMP3:
    num_up_moves: int
    num_down_moves: int

@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):
    alpha3: float = 1.0 # strength of reverse-pull (non-negative value)
    
    def up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(state.num_down_moves / total) if total else 0.5

    def transition(self, state: StateMP3) -> Categorical[StateMP3]:
        up_p = self.up_prob(state)
        return Categorical({
            StateMP3(state.num_up_moves + 1, state.num_down_moves): up_p,
            StateMP3(state.num_up_moves, state.num_down_moves + 1): 1 - up_p
        })


@dataclass
class StockPriceMRP(MarkovRewardProcess[StateMP3]):
    alpha3: float = 1.0 # strength of reverse-pull (non-negative value)
    reward_func: Callable[[StateMP3], float] = lambda _: 0.0

    def up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(state.num_down_moves / total) if total else 0.5

    def transition_reward(self, state: StateMP3)\
            -> Optional[Distribution[Tuple[StateMP3, float]]]:
        
        up_p = self.up_prob(state)
        return Categorical({
            ( StateMP3(state.num_up_moves + 1, state.num_down_moves), self.reward_func(state) ): up_p,
            ( StateMP3(state.num_up_moves, state.num_down_moves + 1), self.reward_func(state) ): 1 - up_p
        })

def traces():
    count_mrp = StockPriceMRP(reward_func = lambda s: s.num_up_moves - s.num_down_moves)
    return count_mrp.reward_traces(Choose([ StateMP3(num_up_moves=0, num_down_moves=0) ]))
