
from enumeration import Movement
from environment import Environment
from exception import SnakeWin
from logger import logger


class Agent:

    def __init__(self, env: Environment):

        self._env = env

    
