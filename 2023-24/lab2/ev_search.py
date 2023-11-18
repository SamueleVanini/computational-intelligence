from typing import Any
import numpy as np
from overrides import override

from scipy.special import softmax


class SelfAdaptiveParams:

    def __init__(self, params: np.ndarray, sigma: np.ndarray, step: int = 1, seed: int = 42) -> None:
        self._generator = np.random.default_rng(seed)
        self._params = np.copy(params)
        self._sigma = sigma
        self._n = step
        self._seed = seed
        self._N = np.random.default_rng(seed).normal

    def __getitem__(self, k: int) -> float:
        return self._params[k]

    @property
    def params(self) -> np.ndarray:
        return self._params
    
    @property
    def sigma(self) -> np.ndarray:
        return self._sigma
    
    @property
    def step(self) -> int:
        return self._n
    
    @property
    def seed(self) -> int:
        return self._seed

    def tweak(self):
        tau = 1 / (self._n ** 0.5)
        sigma = self._sigma * np.exp(tau * self._N(0,1))
        params = self._N(self._params, sigma, self._params.size)
        return SelfAdaptiveParams(params, sigma, self._n+1, self._seed)
    

class ProbabilitySelfAdaptiveParams(SelfAdaptiveParams):

    def __init__(self, probs: np.ndarray, sigma: np.ndarray, step: int = 1, seed: int = 42) -> None:
        super().__init__(params=probs,
                         sigma=sigma,
                         step=step,
                         seed=seed)

    @override
    def tweak(self):
        tau = 1 / (self._n ** 0.5)
        sigma = self._sigma * np.exp(tau * self._N(0,1))
        probs = softmax(self._N(self._params, sigma, self._params.size))
        return ProbabilitySelfAdaptiveParams(probs, sigma, self._n+1, self._seed)