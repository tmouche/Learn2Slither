from enum import Enum

class AgentState(Enum):
    CLUELESS = "CLUELESS"
    IN_TRAINING = "IN_TRAINING"
    TRAINED = "TRAINED"

class Mode(Enum):
    AI_PLAY = "AI_PLAY",
    AI_TRAIN = "AI_TRAIN",
    HUMAIN_PLAY = "HUMAN_PLAY",

class AppStatus(Enum):
    PLAY = "PLAY"
    PAUSE = "PAUSE"
    STOP = "STOP"

class Movement(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3
