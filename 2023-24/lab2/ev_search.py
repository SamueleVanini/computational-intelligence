import numpy as np

from scipy.special import softmax


class SelfAdaptiveParams:

    def __init__(self, probs: np.ndarray, sigma: np.ndarray, step: int = 1, seed: int = 42) -> None:
        self._generator = np.random.default_rng(seed)
        self._probs = probs
        self._sigma = sigma
        self._n = step
        self._seed = seed
        self._N = np.random.default_rng(seed).normal

    @property
    def probs(self) -> np.ndarray:
        return self._probs
    
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
        probs = softmax(self._N(self._probs, sigma, self._probs.size))
        return SelfAdaptiveParams(probs,sigma,self._n+1, self._seed)