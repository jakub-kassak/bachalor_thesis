from typing import cast


class State:
    def __init__(self, i: int, n: int, mc: int):
        self.i = i
        self.n = n
        self.mc = mc

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            return False
        other: State = cast(State, other)
        return self.i == other.i and self.n == other.n and self.mc == other.mc
