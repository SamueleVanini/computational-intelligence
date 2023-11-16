from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from overrides import override
from tqdm import tqdm
from scipy.special import softmax

from strategies import ESStrategy, ProbabilityRules
from typing import Callable, Iterable, Sequence
from game import Match, Nim
from ev_search import SelfAdaptiveParams


class Optimizer(ABC):

    def __init__(self,
                 initial_pop: Sequence[ESStrategy],
                 mu: int,
                 lambd: int,
                 n_generations: int, 
                 fitness: Callable,
                 seed: int = 42) -> None:
        self._mu = mu
        self._lambda = lambd
        self._fitness = fitness
        self._n_gen = n_generations
        self._population = self._evaluator(initial_pop, self._mu)
        self._generator = np.random.default_rng(seed)
        self._seed = seed

    def _evaluator(self, population: Iterable, size: int) -> list[tuple[float, ESStrategy]]:
        return sorted(((self._fitness(ind), ind) for ind in population), key= lambda v: v[0], reverse=True)[:size]

    def evolve(self) -> list[tuple[float, ESStrategy]]:
        best_scores_hist: list[tuple[float, ESStrategy]] = []
        for _ in tqdm(range(self._n_gen)):
            # Parrent selection and genetic operation
            offspring = self._get_offspring()
            # Evaluation
            sorted_offspeang = self._evaluator(offspring, self._lambda)
            # Survival selection and stats
            self._population = self._survival_selection(sorted_offspeang)
            best_scores_hist.append((self._population[0][0], self._population[0][1]))
        return best_scores_hist
    
    def _get_offspring(self) -> Iterable[ESStrategy]:
        idx_pop = self._generator.integers(0, self._mu, size=(self._lambda,))
        offspring_gen = (self._population[i][1] for i in idx_pop)
        return (offspring.tweak() for offspring in offspring_gen)
    
    @abstractmethod
    def _survival_selection(self, sorted_offspeang: list[tuple[float, ESStrategy]]) -> list[tuple[float, ESStrategy]]:
        pass

class CrossoverOptimizer(Optimizer):

    def __init__(self,
                 initial_pop: Sequence[ESStrategy],
                 mu: int,
                 lambd: int,
                 ro: int,
                 n_generations: int, 
                 fitness: Callable,
                 seed: int = 42) -> None:
        super().__init__(initial_pop=initial_pop,
                         mu=mu,
                         lambd=lambd,
                         n_generations=n_generations,
                         fitness=fitness,
                         seed=seed)
        self._ro = ro
    

    @override
    def evolve(self) -> list[tuple[float, ESStrategy]]:
        best_scores_hist: list[tuple[float, ESStrategy]] = []
        for _ in tqdm(range(self._n_gen)):
            # Parrent selection
            p1, p2 = self._tornament(self._ro)[:2]
            # Genetic operation
            crossover = self._generator.choice(np.array([self._one_cut_crossover, self._rand_crossover]))
            individual = crossover(p1[1], p2[1])
            tweaked = (individual.tweak() for _ in range(self._lambda))
            # Evaluation
            sorted_offspeang = self._evaluator(tweaked, self._lambda)
            # Survival selection and stats
            self._population = self._survival_selection(sorted_offspeang)
            best_scores_hist.append((self._population[0][0], self._population[0][1]))
        return best_scores_hist
    
    def _tornament(self, selective_pressure) -> list[tuple[float, ESStrategy]]:
        tornament_pop_idxs = self._generator.choice(range(self._mu), size=(selective_pressure))
        return sorted(((self._fitness(self._population[ind][1]), self._population[ind][1]) for ind in tornament_pop_idxs), key= lambda v: v[0], reverse=True)
    
    def _rand_crossover(self, p1: ESStrategy, p2: ESStrategy) -> ESStrategy:
        parrents = [p1.get_params(), p2.get_params()] 
        n_probs = parrents[0].probs.size
        new_probs = np.zeros(n_probs)
        new_sigma = np.zeros(n_probs)
        for i in range(n_probs):
            parrent_idx = self._generator.choice([0,1])
            new_probs[i] = parrents[parrent_idx].probs[i]
            new_sigma[i] = parrents[parrent_idx].sigma[i]
        params = SelfAdaptiveParams(new_probs, new_sigma, parrents[0].step, parrents[0].seed)
        return ProbabilityRules(p1.get_strategy(), params, p1.seed)
    
    def _one_cut_crossover(self, p1: ESStrategy, p2: ESStrategy) -> ESStrategy:
        possible_cuts = np.arange(1, p1.get_params().probs.size, step=1)
        split = self._generator.choice(possible_cuts)
        p1_params = p1.get_params()
        p2_params = p2.get_params()
        probs = softmax(np.concatenate([p1_params.probs[:split], p2_params.probs[split:]]))
        sigma = np.concatenate([p1_params.sigma[:split], p2_params.sigma[split:]])
        params = SelfAdaptiveParams(probs, sigma, p2_params.step, p2_params.seed)
        return ProbabilityRules(p1.get_strategy(), params, p1.seed)
    

class MuPlusLambda(Optimizer):

    def __init__(self,
                 inital_sol: Sequence[ESStrategy],
                 mu: int,
                 lambd: int,
                 n_genenerations: int,
                 fitness: Callable,
                 seed: int = 42) -> None:
        super().__init__(initial_pop=inital_sol,
                         mu=mu,
                         lambd=lambd,
                         n_generations=n_genenerations,
                         fitness=fitness,
                         seed=seed)
    
    def _survival_selection(self, sorted_offspeang: list[tuple[float, ESStrategy]]) -> list[tuple[float, ESStrategy]]:
        return sorted(self._population + sorted_offspeang, key=lambda v: v[0], reverse=True)[:self._mu]
    
class MuCommaLambda(Optimizer):

    def __init__(self,
                 inital_sol: Sequence[ESStrategy],
                 mu: int,
                 lambd: int,
                 n_genenerations: int,
                 fitness: Callable,
                 seed: int = 42) -> None:
        super().__init__(initial_pop=inital_sol,
                         mu=mu,
                         lambd=lambd,
                         n_generations=n_genenerations,
                         fitness=fitness,
                         seed=seed)
    
    def _survival_selection(self, sorted_offspeang: list[tuple[float, ESStrategy]]) -> list[tuple[float, ESStrategy]]:
        new_pop = (sorted_offspeang + self._population)[:self._mu]
        return sorted(new_pop, key=lambda v: v[0], reverse=True)


class MuRoPlusLambda(CrossoverOptimizer):

    def __init__(self,
                 initial_sol: Sequence[ESStrategy],
                 mu: int,
                 lambd: int,
                 ro: int,
                 n_generations: int,
                 fitness: Callable,
                 seed: int = 42) -> None:
        super().__init__(initial_pop=initial_sol,
                         mu=mu,
                         lambd=lambd, 
                         ro=ro, 
                         n_generations=n_generations, 
                         fitness=fitness, 
                         seed=seed)
        
    @override
    def _survival_selection(self, sorted_offspeang: list[tuple[float, ESStrategy]]) -> list[tuple[float, ESStrategy]]:
        return sorted(self._population + sorted_offspeang, key=lambda v: v[0], reverse=True)[:self._mu]



class MuRoCommaLambda(CrossoverOptimizer):
    
    def __init__(self,
                 initial_sol: Sequence[ESStrategy],
                 mu: int,
                 lambd: int,
                 ro: int,
                 n_generations: int,
                 fitness: Callable,
                 seed: int = 42) -> None:
        super().__init__(initial_pop=initial_sol,
                         mu=mu,
                         lambd=lambd, 
                         ro=ro, 
                         n_generations=n_generations, 
                         fitness=fitness, 
                         seed=seed)

    @override
    def _survival_selection(self, sorted_offspeang: list[tuple[float, ESStrategy]]) -> list[tuple[float, ESStrategy]]:
        new_pop = (sorted_offspeang + self._population)[:self._mu]
        return sorted(new_pop, key=lambda v: v[0], reverse=True)

def fitness(individual: Callable, opponent: Callable, game: Nim, n_games: int = 50) -> float:
    """Evaluate an individual against an opponent in the same game (changing who start first)
    on a multitude of games

    Args:
        individual (Callable): individual to test
        opponent (Callable): opponent to match with the individual
        game (Nim): game choosen to test the match
        n_games (int, optional): Number of game to play. Defaults to 50.

    Returns:
        float: percentage of game won with a certain configuration
    """
    n_wins = 0
    for idx in range(n_games):
        new_game = deepcopy(game)
        if idx % 2 == 0:
            match = Match(new_game, individual, opponent)
            n_wins += (1 - match.play())
        else:
            match = Match(new_game, opponent, individual)
            n_wins += match.play()
    return n_wins / n_games



# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from scipy.special import softmax
#     from functools import partial
#     from game import Nim
#     from strategies import MaxObjectLowestRow, Random, ProbabilityRules

#     nim = Nim(5)
#     probs = softmax(np.random.default_rng(42).uniform(0,1,2))
#     params = SelfAdaptiveParams(probs, np.repeat(0.001, probs.size))
#     es = [ProbabilityRules([MaxObjectLowestRow(), Random()], params)] * 10
#     fit = partial(fitness, opponent=Random(), game=nim)
#     #one_plu_lambda = MuRoPlusLambda(es, lambd=2, ro=len(es)//2, n_generations=(1000 // 10), fitness=fit)
#     one_plu_lambda = MuRoCommaLambda(es, lambd=5, ro=len(es)//2, n_generations=(1000 // 10), fitness=fit)
#     hist = one_plu_lambda.evolve()
#     fig, ax = plt.subplots()
#     hist = np.array(hist)
#     ax.plot(range(one_plu_lambda._n_gen), hist[:, 0])
#     plt.show()