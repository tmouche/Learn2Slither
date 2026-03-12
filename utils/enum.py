from enum import Enum

class AgentState(Enum):
    CLUELESS = "CLUELESS"
    IN_TRAINING = "IN_TRAINING"
    TRAINED = "TRAINED"

class Mode(Enum):
    TRAIN = 0,
    TEST = 1

class Movement(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3