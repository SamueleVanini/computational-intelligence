from copy import deepcopy
import logging
import numpy as np
import matplotlib.pyplot as plt

from game import Match, Nim
from scipy.special import softmax
from functools import partial
from game import Nim
from es import MuPlusLambda, MuCommaLambda, MuRoCommaLambda, MuRoPlusLambda, fitness
from strategies import ExpertMisere, MaxObjectLowestRow, Random, ProbabilityRules, SubtractionExpertMisere
from ev_search import SelfAdaptiveParams

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


def main():
    logging.getLogger().setLevel(logging.INFO)
    MU = 5
    LAMBDA = 20
    RO = 2
    SEED = 42
    N_GAMES = 100

    nim = Nim(5)
    init_pop = create_initial_pop(MU, SEED)
    fit = partial(fitness, opponent=Random(), game=nim)

    # Expert system vs random player
    logging.info(f"Playing: {N_GAMES//2} Expert vs Random and {N_GAMES//2} Random vs Expert")
    wins = sum([1 - Match(deepcopy(nim), ExpertMisere(), Random()).play() for _ in range(N_GAMES // 2)])
    wins += sum([Match(deepcopy(nim), Random(), ExpertMisere()).play() for _ in range(N_GAMES // 2)])
    logging.info(f"% of Expert system vs Random player wins: {wins/N_GAMES}") 

    # Evolution Strategy
    logging.info(f"Running Evolution Strategies with: {init_pop[0].__name__} stategies")
    mu_plus_lambda = MuPlusLambda(init_pop, mu=MU, lambd=LAMBDA, n_genenerations=(5_000 // LAMBDA), fitness=fit)
    mu_comma_lambda = MuCommaLambda(init_pop, mu=MU, lambd=LAMBDA, n_genenerations=(5_000 // LAMBDA), fitness=fit)
    mu_ro_plu_lambda = MuRoPlusLambda(init_pop, mu=MU, lambd=LAMBDA, ro=RO, n_generations=(5_000 // LAMBDA), fitness=fit)
    mu_ro_comma_lambda = MuRoCommaLambda(init_pop, mu=MU, lambd=LAMBDA, ro=RO, n_generations=(5_000 // LAMBDA), fitness=fit)

    logging.info(f"Running: {strings['mu_plus_lambda']}")
    hist_plus = mu_plus_lambda.evolve()
    logging.info(f"Running: {strings['mu_comma_lambda']}")
    hist_comma = mu_comma_lambda.evolve()
    logging.info(f"Running: {strings['mu_ro_plus_lambda']}")
    hist_cross_plus  = mu_ro_plu_lambda.evolve()
    logging.info(f"Running: {strings['mu_ro_comma_lambda']}")
    hist_coss_comma = mu_ro_comma_lambda.evolve()
    
    plotting_data = [
        (mu_plus_lambda, f"{strings['mu_plus_lambda']}" , hist_plus),
        (mu_comma_lambda, f"{strings['mu_comma_lambda']}" , hist_comma),
        (mu_ro_plu_lambda, f"{strings['mu_ro_plus_lambda']}", hist_cross_plus), 
        (mu_ro_comma_lambda, f"{strings['mu_ro_comma_lambda']}", hist_coss_comma)
    ] 
    # plotting
    plotting(2, 2, plotting_data)

def create_initial_pop(size: int, seed: int = 42):
    probs = softmax(np.random.default_rng(seed).uniform(0,1,2))
    params = SelfAdaptiveParams(probs, np.repeat(0.001, probs.size))
    return [ProbabilityRules([MaxObjectLowestRow(), Random()], params)] * size

def plotting(n_rows: int, n_cols: int, data: list):
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
    plt.show()

if __name__ == "__main__":
    main()