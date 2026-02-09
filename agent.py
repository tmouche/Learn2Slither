
from abc import abstractmethod
from enumeration import Movement
from environment import Environment
from exception import SnakeWin
from logger import logger
from typing import List, Dict
from utils import norm

class Agent:

    _env: Environment

    PATH_TO_SAVE: str

    EPOCH: int
    LEARNING_RATE: float
    DISCOUNT: float
    EXPLO_RATE: float
    EXPLO_DECAY: float
    MAX_ACTION: int

    def __init__(self):
        pass

    def __init__(
        self,
        env: Environment,
        epoch: int,
        l_rate: float,
        discount: float,
        e_rate: float,
        e_decay: float,
        max_action: int,
        path_to_save: str | None = None
    ):
        self._env = env

        self.EPOCH = epoch
        self.LEARNING_RATE = l_rate
        self.DISCOUNT = discount
        self.EXPLO_RATE = e_rate
        self.EXPLO_DECAY = e_decay
        self.MAX_ACTION = max_action
        self.PATH_TO_SAVE = path_to_save

        self._RAND_MAX = 10000

    @abstractmethod
    def train():
        pass


    def _retrieve_state_from_env(self) -> List[float]:
        state: List[str] = self._env.snake_view()
        
        state_data: Dict[str, List[int]] = {
            "w_dist": [0, 0, 0, 0],
            "s_dist": [0, 0, 0, 0],
            "ga_dist": [0, 0, 0, 0],
            "ra_dist": [0, 0, 0, 0],
        }

        head_pos: int = self._env.snake.pos[0]
        idx: int = 1
        while True:
            counter: int = 0
            for i in range(4):
                if not state_data['w_dist'][i]:
                    counter += self.__check_state_value(
                        state_data,
                        i,
                        state[int(head_pos + idx * list(self._env.map_move.values())[i])],
                        idx
                    )
            idx += 1
            if not counter:
                break

        normed_state: List[float] = []
        for k in state_data.keys():
            normed_state += norm(state_data[k])
        
        return normed_state 

    def __check_state_value(
            self,
            data: Dict[str, List[int]],
            data_idx: int,
            current: str,
            idx: int
        ):
        match current:
            case 'W':
                data["w_dist"][data_idx] = idx
            case 'S':
                data["s_dist"][data_idx] = idx
            case 'G':
                data["ga_dist"][data_idx] = idx
            case 'R':
                data["ra_dist"][data_idx] = idx
        return 1
                
    def print_map(self, map: List[str], width:int):
        logger.info('\n')
        for i in range(len(map)):
            print(map[i],end="")
            if not (i+1) % width and i:
                print()
        print()

