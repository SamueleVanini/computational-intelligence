from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from operator import xor
from scipy.special import softmax
from typing import Callable, Sequence
from overrides import override
from math import floor
from ev_search import ProbabilitySelfAdaptiveParams, SelfAdaptiveParams

import numpy as np

from game import GameType, Nim, Nimply

class Strategy(ABC):

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._generator = np.random.default_rng(seed)

    @property
    def seed(self) -> int:
        return self._seed

    @abstractmethod
    def __call__(self, state: Nim) -> Nimply:
        pass

class EStrategy(Strategy):
    
    @abstractmethod
    def tweak(self):
        pass

    @abstractmethod
    def get_params(self) -> SelfAdaptiveParams:
        pass

class ESPlayer:

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._generator = np.random.default_rng(seed)

    @property
    def seed(self) -> int:
        return self._seed
    
    @abstractmethod
    def __call__(self, state: Nim) -> Nimply:
        pass

    @abstractmethod
    def tweak(self):
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
    
class PredeterminedRules(Strategy):
    """
    Strategy where a mix of hard-coded rules are evaluate, 
    it uses MaxObjectLowestRow for the general case
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)

    def __call__(self, state: Nim) -> Nimply:
        if state.type is not GameType.MISERE:
            raise Exception("This strategy is not implement for normal game")
        not_empty_heaps = [(i, n_obj) for i, n_obj in enumerate(state.rows) if n_obj > 0]
        n_not_empty_heaps = len(not_empty_heaps)
        if n_not_empty_heaps == 1:
            # If there is only 1 heap, checks the number of objects to choose how many remove
            heap = not_empty_heaps[0]
            remaining_obj = heap[1] - min(heap[1] - 1, state.k)
            if remaining_obj % 2 == 0:
                return Nimply(heap[0], max(1, heap[1] - remaining_obj - 1))
            else:
                return Nimply(heap[0], max(1, heap[1] - remaining_obj))
        elif n_not_empty_heaps == 2:
            idx_max, max_obj = max(not_empty_heaps, key=lambda v: v[1])
            _, min_obj = min(not_empty_heaps, key=lambda v: v[1])
            # I want to always have a odd number of objects after the move
            # If both heaps have only 1 element I will lose but I'm forced to take at least an element
            if (max_obj - min_obj) % 2 == 0:
                return Nimply(idx_max, max(1, min(max_obj - min_obj - 1, state.k)))
            else:
                return Nimply(idx_max, max(1, min(max_obj - min_obj, state.k)))
        else:
            return MaxObjectLowestRow(self._seed)(state)


class AdaptiveHeapRule(EStrategy):

    def __init__(self, heap: int, params: SelfAdaptiveParams|None = None, seed: int = 42) -> None:
        super().__init__(seed)
        self._heap = heap
        if params is None:
            self._params = SelfAdaptiveParams(((heap*2+1) - 1) * self._generator.random(size=4) + 1, np.full(shape=(4,), fill_value=1, dtype=float))
        else:
            self._params = params
        self._max = float(self._params[0])
        self._min = float(self._params[1])
        self._n_obj = round(self._params[2])
        self._priority = float(self._params[3])

    @property
    def priority(self) -> float:
        return self._priority

    def is_valid(self, state: Nim) -> bool:
        nobj_heap = state.rows[self._heap]
        return nobj_heap > 0 and floor(self._min) <= nobj_heap <= round(self._max)
    
    def __call__(self, state: Nim) -> Nimply:
        nobj_heap = state.rows[self._heap]
        return Nimply(self._heap, min(nobj_heap, max(1, self._n_obj), state.k))
    
    def tweak(self):
        return AdaptiveHeapRule(self._heap, self._params.tweak(), self._seed)
    
    def get_params(self) -> SelfAdaptiveParams:
        return self._params
    
class AdaptivePlay(ESPlayer):

    def __init__(self, 
                 nrow: int, 
                 heap_rules: Sequence[AdaptiveHeapRule]|None = None, 
                 seed: int = 42) -> None:
        super().__init__(seed)
        self._nrows = nrow
        if heap_rules is None:
            self._heap_rules = [AdaptiveHeapRule(heap) for heap in range(nrow) for _ in range(heap + 4)]
        else:
            self._heap_rules = heap_rules
        self.__name__ = "AdaptiveRule"

    def tweak(self):
        tweaked_rules = [heap_rule.tweak() for heap_rule in self._heap_rules]
        return AdaptivePlay(self._nrows, heap_rules=tweaked_rules) # type: ignore
    
    def __call__(self, state: Nim) -> Nimply:
        valid_rules = [rule for rule in self._heap_rules if rule.is_valid(state)]
        if len(valid_rules) != 0:
            priority, rule = max(((rule.priority, rule) for rule in valid_rules), key=lambda v: v[0])
            return rule(state)
        r = next((r for r, o in enumerate(state.rows) if o), None)
        return Nimply(r, 1)
    
    @property
    def heap_rules(self) -> Sequence[AdaptiveHeapRule]:
        return self._heap_rules
    
    @property
    def nrow(self) -> int:
        return self._nrows


class ProbabilityRules(ESPlayer):

    def __init__(self, 
                 strategies: Sequence[Callable],
                 params: ProbabilitySelfAdaptiveParams,
                 seed: int = 42) -> None:
        super().__init__(seed)
        self._strategies = strategies
        self._n_strategies = len(strategies)
        assert self._n_strategies == params.params.size
        self._params = params
        self._generator = np.random.default_rng(seed)
        self.__name__ = f"{[strat.__class__.__name__ for strat in strategies]}"

    def get_strategy(self) -> Sequence[Callable]:
        return self._strategies
    
    def get_params(self) -> ProbabilitySelfAdaptiveParams:
        return self._params

    @override
    def __call__(self, state: Nim) -> Nimply:
        idx = self._generator.choice(range(self._n_strategies), p=self._params.params)
        return self._strategies[idx](state)
    
    @override
    def tweak(self):
        return ProbabilityRules(self._strategies, self._params.tweak(), self._seed)