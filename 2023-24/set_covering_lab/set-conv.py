from math import ceil
from gx_utils import PriorityQueue

from functools import reduce
from random import random, seed
from typing import Callable
from tqdm.auto import tqdm

import numpy as np
from state import State

seed(42)
DEBUG = False
PROBLEM_SIZE = 50
NUM_SETS = 100
SETS = tuple(np.array([random() < 0.3 for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS))


def goal_check(state) -> bool:
    return np.all(
        reduce(
            np.logical_or,
            [SETS[i] for i in state.taken],
            np.array([False for _ in range(PROBLEM_SIZE)]),
        )
    )


def distance(state):
    return PROBLEM_SIZE - sum(
        reduce(
            np.logical_or,
            [SETS[i] for i in state.taken],
            np.array([False for _ in range(PROBLEM_SIZE)]),
        )
    )


def actual_cost(state):
    return len(state.taken)


def h1(state):
    largest_set_size = max(sum(s) for s in SETS)
    missing_size = PROBLEM_SIZE - sum(covered(state))
    optimistic_estimate = ceil(missing_size / largest_set_size)
    return optimistic_estimate


def h2(state):
    already_covered = covered(state)
    if np.all(already_covered):
        return 0
    largest_set_size = max(sum(np.logical_and(s, np.logical_not(already_covered))) for s in SETS)
    missing_size = PROBLEM_SIZE - sum(already_covered)
    optimistic_estimate = ceil(missing_size / largest_set_size)
    return optimistic_estimate


def h3(state):
    already_covered = covered(state)
    if np.all(already_covered):
        return 0
    missing_size = PROBLEM_SIZE - sum(already_covered)
    candidates = sorted((sum(np.logical_and(s, np.logical_not(already_covered))) for s in SETS), reverse=True)
    taken = 1
    while sum(candidates[:taken]) < missing_size:
        taken += 1
    return taken


def h4(state):
    """h() heauristic for A* algorithm, it use the average lenght of set that don't already cover
    the the problem space. In the worst case it could led to a greddy best-first algorithm (the function could
    overstimate the cost to reach the destination).

    Args:
        state (State): current state

    Returns:
        int: estimate of the cost to reach a solution
    """
    already_covered = covered(state)
    if np.all(already_covered):
        return 0
    cum_sum = 0
    c = 0
    for s in SETS:
        cum_sum += sum(np.logical_and(s, np.logical_not(already_covered)))
        c += 1
    mean_set_size = cum_sum / (c - 1)
    missing_size = PROBLEM_SIZE - sum(already_covered)
    optimistic_estimate = ceil(missing_size / mean_set_size)
    return optimistic_estimate


def covered(state):
    return reduce(
        np.logical_or,
        [SETS[i] for i in state.taken],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    )


def search(initial_state: State, goal_check: Callable, priotity_function: Callable):

    frontier = PriorityQueue()
    counter = 0
    current_state = initial_state
    with tqdm(total=None) as pbar:
        while not goal_check(current_state):
            counter += 1
            for action in current_state.not_taken:
                new_state = State(
                    sets_taken=current_state.taken ^ {action}, sets_not_taken=current_state.not_taken ^ {action}
                )
                p = priotity_function(new_state)
                frontier.push(new_state, p=p)
            if frontier:
                current_state = frontier.pop()
            else:
                return None, -1
            pbar.update(1)
    return current_state, counter


def debug():
    initial_state = State(sets_taken=set(), sets_not_taken=set(range(NUM_SETS)))

    print("Starting searching")

    euristics = [distance, h2, h3, h4]

    for e in euristics:
        final_state, steps = search(
            initial_state=initial_state,
            goal_check=goal_check,
            priotity_function=lambda state: actual_cost(state) + e(state),
        )
        print(f"Solved with {e.__name__} in {steps} steps, {len(final_state.taken)} tiles")
        print(f"Final state: {final_state}")
    for s in SETS:
        print(s)


if __name__ == "__main__":
    if DEBUG:
        debug()
    else:
        initial_state = State(sets_taken=set(), sets_not_taken=set(range(NUM_SETS)))

        print("Starting searching")

        final_state, steps = search(
            initial_state=initial_state,
            goal_check=goal_check,
            # priotity_function=lambda state: actual_cost(state) + distance(state),
            priotity_function=lambda state: actual_cost(state) + h4(state),
        )

        if final_state is not None:
            print(f"Distance from goal: {distance(final_state)}")
            print(f"Final state: {final_state}")
            print(f"Solved in {steps} steps, {len(final_state.taken)} tiles")
        else:
            print("Problem not solvable")
