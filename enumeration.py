from enum import Enum

class Movement(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3

class Mode(Enum):
    TRAIN = 0,
    TEST = 1