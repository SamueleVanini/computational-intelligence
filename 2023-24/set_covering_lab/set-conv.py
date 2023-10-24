from gx_utils import PriorityQueue

from functools import reduce
from random import random, seed
from typing import Callable

import numpy as np
from state import State

seed(42)
PROBLEM_SIZE = 5
NUM_SETS = 10
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


def search(initial_state: State, goal_check: Callable, priotity_function: Callable):

    frontier = PriorityQueue()
    counter = 0
    current_state = initial_state
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
    return current_state, counter


if __name__ == "__main__":
    a = set(range(NUM_SETS))

    initial_state = State(sets_taken=set(), sets_not_taken=set(range(NUM_SETS)))

    print("Starting searching")

    final_state, steps = search(
        initial_state=initial_state,
        goal_check=goal_check,
        priotity_function=lambda state: actual_cost(state) + distance(state),
    )

    if final_state is not None:
        print(f"Distance from goal: {distance(final_state)}")
        print(f"Final state: {final_state}")
        print(f"Solved in {steps} steps, {len(final_state.taken)} tiles")
    else:
        print("Problem not solvable")
