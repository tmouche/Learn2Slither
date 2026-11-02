from abc import abstractmethod
from core.environment import Environment
from datetime import datetime
from utils.logger import logger
from typing import List, Dict, Generator, Optional
from utils.dataclass import AgentSettings
from utils.enum import Movement
from utils.exception import UnimplementedError
from utils.types import FloatT
from utils.utils_file import norm

class Agent:
    _env: Environment

    _name: str

    PATH_TO_SAVE: str

    _epoch_max: int
    _learning_rate: float
    _discount: float
    _explo_rate: float
    _explo_decay: float
    _max_action: int

    _reward_green_apple: FloatT
    _reward_lose: FloatT
    _reward_nothing: FloatT
    _reward_red_apple: FloatT
    _reward_win: FloatT

    def __init__(
        self,
        env: Environment,
        agent_settings: AgentSettings,
        name: Optional[str] = None
    ):
        self._env = env
        self._discount = agent_settings.hyper_params.discount
        self._epoch_max = agent_settings.hyper_params.epoch
        self._explo_decay = agent_settings.hyper_params.exploration_decay
        self._explo_rate = agent_settings.hyper_params.exploration_rate
        self._learning_rate = agent_settings.hyper_params.learning_rate
        self._max_action = agent_settings.hyper_params.max_action
        self._reward_green_apple = agent_settings.rewards.green_apple
        self._reward_lose = agent_settings.rewards.loose
        self._reward_nothing = agent_settings.rewards.noting
        self._reward_red_apple = agent_settings.rewards.red_apple
        self._reward_win = agent_settings.rewards.win
        self._name = agent_settings.agent_name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def train(self):
        return UnimplementedError(f'{type(self).__name__}.train')

    def train_sequentially(self) -> Generator:
        return UnimplementedError(f'{type(self).__name__}.train_sequentially')

    @abstractmethod
    def load(self):
        return UnimplementedError(f'{type(self).__name__}.load')

    @abstractmethod
    def save(self):
        return UnimplementedError(f'{type(self).__name__}.save')

    @abstractmethod
    def take_action(self) -> Movement:
        return UnimplementedError(f'{type(self).__name__}.take_action')

    def save_training_metrics(self):
        return UnimplementedError(f'{type(self).__name__}.save_training_metrics')

    def plot_training_metrics(self,):
        return UnimplementedError(f'{type(self).__name__}.plot_training_metrics')

    def _retrieve_state_from_env(self) -> List[float]:
        state_data: Dict[str, List[int]] = self._retrieve_state_data()
        normed_state: List[float] = []
        for v in state_data.values():
            for i in range(4):
                if i <= 1:
                    normed_state.append(round(v[i]/self._env.height, 1))
                else:
                    normed_state.append(round(v[i]/self._env.width, 1))
        return normed_state

    def _retrieve_state_data(self) -> Dict[str, List[int]]:
        state: List[str] = self._env.snake_view()
        state_data: Dict[str, List[int]] = {
            "d_dist": [0, 0, 0, 0],
            "ga_dist": [0, 0, 0, 0],
            "ra_dist": [0, 0, 0, 0],
        }
        head_pos: int = self._env.snake.pos[0]
        idx: int = 1
        completed: int = 0
        while completed < 4:
            for i in range(4):
                if self._is_key_threated(state_data, i) == True:
                    continue
                if self._check_state_value(
                    state_data,
                    i,
                    state[int(head_pos + idx * list(self._env.map_move.values())[i])],
                    idx,
                ) == True:
                    completed += 1
            idx += 1
        return state_data

    @staticmethod
    def _is_key_threated(state_data: Dict[str, List[int]], idx: int):
        stop: bool = False
        for key in state_data.keys():
            if state_data[key][idx]:
                stop = True
                break
        return stop

    def _check_state_value(
            self,
            data: Dict[str, List[int]],
            data_idx: int,
            current: str,
            idx: int
        ):
        match current:
            case 'W' | 'S':
                data['d_dist'][data_idx] = idx
            case 'G':
                data['ga_dist'][data_idx] = idx
            case 'R':
                data['ra_dist'][data_idx] = idx
            case _:
                return False
        return True
                
    def print_map(self, map: List[str], width:int):
        logger.info('\n')
        for i in range(len(map)):
            print(map[i],end="")
            if not (i+1) % width and i:
                print()
        print()

    def get_map(self) -> List[str]:
        return self._env.map