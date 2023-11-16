from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
import random
from operator import xor
from typing import Callable, Collection, List, Sequence
from overrides import override
from scipy.special import softmax
from ev_search import SelfAdaptiveParams

import numpy as np
from numpy.typing import ArrayLike

from game import Nim, Nimply

class Strategy(ABC):

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._generator = np.random.default_rng(seed)

    @abstractmethod
    def __call__(self, state: Nim) -> Nimply:
        pass

class ESStrategy(Strategy):

    @property
    def seed(self) -> int:
        return self._seed

    @abstractmethod
    def __call__(self, state: Nim) -> Nimply:
        return super().__call__(state)
    
    @abstractmethod
    def tweak(self):
        pass

    @abstractmethod
    def get_strategy(self) -> Sequence[Callable]:
        pass

    @abstractmethod
    def get_params(self) -> SelfAdaptiveParams:
        pass

class ExpertMisere(Strategy):

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)

    def __call__(self, state: Nim) -> Nimply:
        nim_sum = self._nim_sum(state)
        ply = self._normal_play(state, nim_sum)
        if self._check_misere(state):
            ply = self._misere_play(state)
        return ply
    
    def _nim_sum(self, state: Nim) -> int:
        return reduce(xor, state.rows)

    def _normal_play(self, state: Nim, nim_sum: int) -> Nimply:
        if nim_sum != 0:
            for r, o in enumerate(state.rows):
                if self._check_balance_move(o, nim_sum):
                    return Nimply(r, o - xor(o, nim_sum))
        r = next((r for r, o in enumerate(state.rows) if o), None)
        return Nimply(r, 1)
    
    def _check_balance_move(self, num_object: int, nim_sum: int) -> bool:
        return xor(num_object, nim_sum) < num_object

    def _check_misere(self, state: Nim) -> bool:
        return sum(map(lambda x: x > 1, state.rows)) == 1
    
    def _misere_play(self, state: Nim) -> Nimply:
        idx, num_object = next(((i, o) for i, o in enumerate(state.rows) if o > 1))
        if sum(map(lambda x: x > 0, state.rows)) % 2 == 0:
            return Nimply(idx, num_object)
        return Nimply(idx, num_object-1)
    
class SubtractionExpertMisere(ExpertMisere):

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)

    def __call__(self, state: Nim) -> Nimply:
        reduced_heaps = deepcopy(state)
        reduced_heaps._rows = [x % (reduced_heaps.k + 1) for x in state.rows]
        nim_sum = self._nim_sum(reduced_heaps)
        ply = self._normal_play(reduced_heaps, nim_sum)
        if self._check_subtraction_misere(reduced_heaps):
            row = self._generator.choice([r for r, c in enumerate(state.rows) if c > 0])
            #row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
            ply = Nimply(row, reduced_heaps.k)
        elif self._check_misere(reduced_heaps):
            ply = self._misere_play(reduced_heaps)
        return ply

    def _check_subtraction_misere(self, state: Nim) -> bool:
        return sum(map(lambda x: x != 0, state.rows)) == 0


class Random(Strategy):

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self.__name__ = "Random"

    def __call__(self, state: Nim) -> Nimply:
        row = self._generator.choice([r for r, c in enumerate(state.rows) if c > 0])
        num_objects = self._generator.integers(1, min(state.rows[row], state.k), dtype=int, endpoint=True)
        return Nimply(row, num_objects)
    
class MaxObjectLowestRow(Strategy):

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self.__name__ = "MaxObjectLowestRow"

    def __call__(self, state: Nim) -> Nimply:
        """Pick always the maximum possible number of the lowest row"""
        possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, min(c + 1, state.k + 1))]
        return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))

class ProbabilityRules(ESStrategy):

    def __init__(self, 
                 strategies: Sequence[Callable],
                 params: SelfAdaptiveParams,
                 seed: int = 42) -> None:
        super().__init__(seed)
        self._strategies = strategies
        self._n_strategies = len(strategies)
        assert self._n_strategies == params.probs.size
        self._params = params
        self._generator = np.random.default_rng(seed)
        self.__name__ = f"{[strat.__name__ for strat in strategies]}"

    @override
    def get_strategy(self) -> Sequence[Callable]:
        return self._strategies
    
    @override
    def get_params(self) -> SelfAdaptiveParams:
        return self._params

    @override
    def __call__(self, state: Nim) -> Nimply:
        idx = self._generator.choice(range(self._n_strategies), p=self._params.probs)
        return self._strategies[idx](state)
    
    @override
    def tweak(self):
        return ProbabilityRules(self._strategies, self._params.tweak(), self._seed)

def adaptive(state: Nim) -> Nimply:
    """A strategy that can adapt its parameters"""
    genome = {"love_small": 0.5}
    raise NotImplementedError()