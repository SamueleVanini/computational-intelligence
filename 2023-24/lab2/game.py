from collections import namedtuple
from enum import Enum
from functools import reduce
import logging
from operator import xor
from typing import Callable

# Tuple that rappresent a move in the game (row in which I'm taking objects and how many I'm taking)
Nimply = namedtuple("Nimply", "row, num_objects")

class GameType(Enum):
    NORMAL = 0,
    MISERE = 1

class Nim:
    def __init__(self, num_rows: int, k: int|None = None, type: GameType = GameType.MISERE) -> None:
        """Game's state of Nim

        Args:
            num_rows (int): number of rows in the game
            k (int, optional): Maximum number of object that can be removed in 1 turn. Defaults to None.
        """
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k if k is not None else self._rows[-1]
        self._type = type

    def __bool__(self) -> bool:
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(self._rows)
    
    @property
    def k(self) -> int:
        return self._k
    
    @property
    def type(self) -> GameType:
        return self._type

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects

class Match:

    def __init__(self, game: Nim, player1: Callable, player2: Callable) -> None:
        self._game = game
        self._player = [player1, player2]

    def play(self) -> int:
        """Run a match between 2 player

        Returns:
            int: return the index of the winning player
        """
        player = 0
        logging.debug(f"Game starting - status: {self._game} - nim_sum: {self._nim_sum(self._game)}")
        while self._game:
            ply = self._player[player](self._game)
            logging.debug(f"ply: player {player} plays {ply}")
            self._game.nimming(ply)
            logging.debug(f"status: {self._game} - nim_sum: {self._nim_sum(self._game)}")
            player = 1 - player
        match self._game.type:
            case GameType.NORMAL:
                return 1 - player
            case GameType.MISERE:
                return player
            
    def _nim_sum(self, state: Nim):
        return reduce(xor, state.rows)