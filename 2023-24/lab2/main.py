from copy import deepcopy
import itertools
import logging
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

from game import Match, Nim
from scipy.special import softmax
from functools import partial
from es import MuPlusLambda, MuCommaLambda, MuRoCommaLambda, MuRoPlusLambda, fitness, prob_one_cut_crossover, prob_rand_crossover, adaptive_one_cut_crossover, adaptive_rand_crossover
from strategies import AdaptivePlay, ExpertMisere, MaxObjectLowestRow, PredeterminedRules, Random, ProbabilityRules, Strategy, SubtractionExpertMisere
from ev_search import ProbabilitySelfAdaptiveParams

DEBUG = False

greek_symbols = {
    "mu": "\u03BC",
    "lambda": "\u03BB",
    "rho": "\u03C1"
}

strings = {
    "mu_plus_lambda": f"({greek_symbols['mu']}+{greek_symbols['lambda']})",
    "mu_comma_lambda": f"({greek_symbols['mu']},{greek_symbols['lambda']})",
    "mu_ro_plus_lambda": f"({greek_symbols['mu']}/{greek_symbols['rho']}+{greek_symbols['lambda']})",
    "mu_ro_comma_lambda": f"({greek_symbols['mu']}/{greek_symbols['rho']},{greek_symbols['lambda']})"
}


def main(MU: int, LAMBDA: int, RO: int, SEED: int, N_GAMES: int, N_GEN: int):
    
    nim = Nim(5)
    opponent = create_prob_initial_pop([Random(), PredeterminedRules(), ExpertMisere()], 1)[0]
    fit = partial(fitness, opponent=opponent, game=nim, n_games=20)
    generator = np.random.default_rng(SEED)
    prob_one_cut = partial(prob_one_cut_crossover, generator=generator)
    prob_rand = partial(prob_rand_crossover, generator=generator)
    adapt_one_cut = partial(adaptive_one_cut_crossover, generator=generator)
    adapt_rand = partial(adaptive_rand_crossover, generator=generator)

    # 1 vs 1 games
    players = [PredeterminedRules(), ExpertMisere(), Random(), MaxObjectLowestRow()]
    matchs = itertools.combinations(players, 2)

    for m in matchs:
        logging.info(f"Playing: {N_GAMES//2} {m[0].__class__.__name__} vs {m[1].__class__.__name__} and {N_GAMES//2} {m[1].__class__.__name__} vs {m[0].__class__.__name__}")
        wins = sum([1 - Match(deepcopy(nim), m[0], m[1]).play() for _ in range(N_GAMES // 2)])
        wins += sum([Match(deepcopy(nim), m[1], m[0]).play() for _ in range(N_GAMES // 2)])
        logging.info(f"% of {m[0].__class__.__name__} victory: {wins/N_GAMES}")

    # Evolution Strategy
    probs_strategies_to_test = [[PredeterminedRules(), MaxObjectLowestRow(), Random()]]
    
    plotting_data = []
    plotting_data.append(evolve_prob_player(MU, LAMBDA, RO, N_GEN, probs_strategies_to_test, fit, prob_one_cut, prob_rand))
    plotting_data.append(evolve_adapt_player(MU, LAMBDA, RO, N_GEN, len(nim.rows), fit, adapt_one_cut, adapt_rand))
        
    # plotting
    for title, data in plotting_data:
        plotting(2, 2, data, fig_title=title)

def create_prob_initial_pop(strategies: Sequence[Strategy], size: int, seed: int = 42):
    probs = softmax(np.random.default_rng(seed).uniform(0,1,len(strategies)))
    params = ProbabilitySelfAdaptiveParams(probs, np.repeat(0.1, probs.size))
    return [ProbabilityRules(strategies, params)] * size

def create_adapt_initial_pop(nrow:int, size: int, seed = 42):
    return [AdaptivePlay(nrow, seed=seed)] * size

def evolve_prob_player(MU, LAMBDA, RO, N_GEN, probs_strategies_to_test, fit, prob_one_cut, prob_rand):
    
    plotting_data = []
    
    for strategies in probs_strategies_to_test:

        init_pop = create_prob_initial_pop(strategies, MU, SEED)
        
        logging.info(f"Running Evolution Strategies with: {init_pop[0].__name__} stategies")
        mu_plus_lambda = MuPlusLambda(init_pop, mu=MU, lambd=LAMBDA, n_genenerations=N_GEN, fitness=fit)
        mu_comma_lambda = MuCommaLambda(init_pop, mu=MU, lambd=LAMBDA, n_genenerations=N_GEN, fitness=fit)
        mu_ro_plu_lambda = MuRoPlusLambda(init_pop, mu=MU, lambd=LAMBDA, ro=RO, n_generations=N_GEN, fitness=fit, one_cut_crossover=prob_one_cut, rand_crossover=prob_rand)
        mu_ro_comma_lambda = MuRoCommaLambda(init_pop, mu=MU, lambd=LAMBDA, ro=RO, n_generations=N_GEN, fitness=fit, one_cut_crossover=prob_one_cut, rand_crossover=prob_rand)

        logging.info(f"Running: {strings['mu_plus_lambda']}")
        hist_plus = mu_plus_lambda.evolve()
        logging.info(f"Best fitness: {hist_plus[-1][0]}, weights: {hist_plus[-1][1].get_params().params}") # type: ignore
        
        logging.info(f"Running: {strings['mu_comma_lambda']}")
        hist_comma = mu_comma_lambda.evolve()
        best_individual = max(((result[0], result[1]) for result in hist_comma), key=lambda v: v[0])
        logging.info(f"Best fitness: {best_individual[0]}, weights: {best_individual[1].get_params().params}") # type: ignore
        
        logging.info(f"Running: {strings['mu_ro_plus_lambda']}")
        hist_cross_plus  = mu_ro_plu_lambda.evolve()
        logging.info(f"Best fitness: {hist_cross_plus[-1][0]}, weights: {hist_cross_plus[-1][1].get_params().params}") # type: ignore
        
        logging.info(f"Running: {strings['mu_ro_comma_lambda']}")
        hist_cross_comma = mu_ro_comma_lambda.evolve()
        best_individual = max(((result[0], result[1]) for result in hist_cross_comma), key=lambda v: v[0])
        logging.info(f"Best fitness: {best_individual[0]}, weights: {best_individual[1].get_params().params}") # type: ignore
        
        plotting_data.append((init_pop[0].__name__, [
            (mu_plus_lambda, f"{strings['mu_plus_lambda']}" , hist_plus),
            (mu_comma_lambda, f"{strings['mu_comma_lambda']}" , hist_comma),
            (mu_ro_plu_lambda, f"{strings['mu_ro_plus_lambda']}", hist_cross_plus), 
            (mu_ro_comma_lambda, f"{strings['mu_ro_comma_lambda']}", hist_cross_comma)
        ]))
    return plotting_data

def evolve_adapt_player(MU, LAMBDA, RO, N_GEN, nrow, fit, adapt_one_cut, adapt_rand):
    init_pop = create_adapt_initial_pop(nrow, MU, SEED)
        
    logging.info(f"Running Evolution Strategies with: {init_pop[0].__name__} stategies")
    mu_plus_lambda = MuPlusLambda(init_pop, mu=MU, lambd=LAMBDA, n_genenerations=N_GEN, fitness=fit)
    mu_comma_lambda = MuCommaLambda(init_pop, mu=MU, lambd=LAMBDA, n_genenerations=N_GEN, fitness=fit)
    mu_ro_plu_lambda = MuRoPlusLambda(init_pop, mu=MU, lambd=LAMBDA, ro=RO, n_generations=N_GEN, fitness=fit, one_cut_crossover=adapt_one_cut, rand_crossover=adapt_rand)
    mu_ro_comma_lambda = MuRoCommaLambda(init_pop, mu=MU, lambd=LAMBDA, ro=RO, n_generations=N_GEN, fitness=fit, one_cut_crossover=adapt_one_cut, rand_crossover=adapt_rand)
    logging.info(f"Running: {strings['mu_plus_lambda']}")
    hist_plus = mu_plus_lambda.evolve()
    logging.info(f"Best fitness: {hist_plus[-1][0]}")
    
    logging.info(f"Running: {strings['mu_comma_lambda']}")
    hist_comma = mu_comma_lambda.evolve()
    best_individual = max(result[0] for result in hist_comma)
    logging.info(f"Best fitness: {best_individual}")
    
    logging.info(f"Running: {strings['mu_ro_plus_lambda']}")
    hist_cross_plus  = mu_ro_plu_lambda.evolve()
    logging.info(f"Best fitness: {hist_cross_plus[-1][0]}")
    
    logging.info(f"Running: {strings['mu_ro_comma_lambda']}")
    hist_cross_comma = mu_ro_comma_lambda.evolve()
    best_individual = max(result[0] for result in hist_cross_comma)
    logging.info(f"Best fitness: {best_individual}")
    
    return (init_pop[0].__name__, [
        (mu_plus_lambda, f"{strings['mu_plus_lambda']}" , hist_plus),
        (mu_comma_lambda, f"{strings['mu_comma_lambda']}" , hist_comma),
        (mu_ro_plu_lambda, f"{strings['mu_ro_plus_lambda']}", hist_cross_plus), 
        (mu_ro_comma_lambda, f"{strings['mu_ro_comma_lambda']}", hist_cross_comma)
    ])

def plotting(n_rows: int, n_cols: int, data: list, fig_title: str):
    assert n_rows * n_cols == len(data), f"Expected row*col: {n_rows * n_cols}, but got {len(data)}"
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            opt_object = data[i][0]
            title = data[i][1]
            hist = np.array(data[i][2])
            ax[row, col].plot(range(opt_object._n_gen), hist[:, 0])
            ax[row, col].title.set_text(title)
            ax[row, col].set_xlabel("Generations")
            ax[row, col].set_ylabel("% of victories")
            i += 1
    fig.tight_layout(pad=1.0)
    fig.suptitle(fig_title)
    plt.show()

if __name__ == "__main__":
    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
        MU = 5
        LAMBDA = 50
        RO = 4
        SEED = 42
        N_GAMES = 2
        N_GEN = 1000 // LAMBDA
    else:
        logging.getLogger().setLevel(logging.INFO)
        MU = 5
        LAMBDA = 50
        RO = 4
        SEED = 42
        N_GAMES = 100
        N_GEN = 10_000 // LAMBDA
    main(MU, LAMBDA, RO, SEED, N_GAMES, N_GEN)