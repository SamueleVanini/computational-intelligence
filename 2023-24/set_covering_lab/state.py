from typing import Set


class State:
    def __init__(self, sets_not_taken: Set, sets_taken: Set = set()) -> None:
        self._sets_taken = sets_taken
        self._sets_not_taken = sets_not_taken

    @property
    def taken(self) -> Set:
        return self._sets_taken

    @property
    def not_taken(self) -> Set:
        return self._sets_not_taken
    
    def __lt__(self, other) -> bool:
        return len(self._sets_taken) < len(other.taken)
    
    def __str__(self) -> str:
        return f"taken = {self._sets_taken} not_taken = {self._sets_not_taken}"