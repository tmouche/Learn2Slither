from abc import abstractmethod
from core.environment import Environment
from datetime import datetime
from math import log
from mlp.utils.types import ArrayF, FloatT
from numpy import mean, ndarray
from pandas import DataFrame, Series, read_csv
from plotly.graph_objects import Figure
from secrets import randbelow
from typing import List, Dict, Generator, Optional, Tuple
from utils.dataclass import AgentSettings
from utils.constant import EPS, RAND_MAX
from utils.enum import Movement
from utils.exception import (
    SnakeGreenApple,
    SnakeLoose,
    SnakeLooseTail,
    SnakeLooseWall,
    SnakeRedApple,
    SnakeWin,
    UnimplementedError,
)
from utils.logger import logger
from utils.utils_file import create_folder, hash_list, prep_to_dict
import os
import plotly.express as px

class Agent:
    SAVE_FOLDER = "models"
    PLOT_FOLDER = "plots"
    HISTORY_FILE = "history.csv"
    HISTORY_COLUMNS = [
        "model",
        "discount",
        "exploration_rate",
        "learning_rate",
        "max_action",
        "epoch",
        "green_apple_reward",
        "win_reward",
        "red_apple_malus",
        "nothing_malus",
        "loose_malus",
        "num_snake_hit",
        "num_wall_hit",
        "snake_mean_rewards",
        "snake_max_rewards",
        "snake_mean_len",
        "snake_max_len",
    ]

    _env: Environment
    _name: str

    # Fixed parameters
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

    # Epoch Functional
    _epoch: int
    _state: List[FloatT]
    _numerical_state: FloatT
    _explo_rate_in_use: FloatT
    _action: int
    _done: bool

    # Epoch Stat
    _acc_reward: FloatT
    _green_apple: int
    _red_apple: int

    # Training Stat
    _green_apple_history: List[int]
    _red_apple_history: List[int]
    _reward_history: List[float]
    _snake_len_history: List[int]
    _snake_hit: List[int]
    _wall_hit: List[int]

    _movement_default: List[int]

    _base: List[int]
    
    def __init__(
        self,
        env: Environment,
        agent_settings: AgentSettings,
    ):
        self._env = env
        if agent_settings.hyper_params is not None:
            self._discount = agent_settings.hyper_params.discount
            self._epoch_max = agent_settings.hyper_params.epoch
            self._explo_decay = agent_settings.hyper_params.exploration_decay
            self._explo_rate = agent_settings.hyper_params.exploration_rate
            self._learning_rate = agent_settings.hyper_params.learning_rate
            self._max_action = agent_settings.hyper_params.max_action
        if agent_settings.rewards is not None:
            self._reward_green_apple = agent_settings.rewards.green_apple
            self._reward_lose = agent_settings.rewards.loose
            self._reward_nothing = agent_settings.rewards.noting
            self._reward_red_apple = agent_settings.rewards.red_apple
            self._reward_win = agent_settings.rewards.win
        self._name = agent_settings.agent_name
        self._base = []
        self._acc_reward = 0.
        if agent_settings.hyper_params is not None:
            self._base = [0 for _ in range(self._epoch_max)]
        self._movement_default = range(len(Movement))

    @property
    def name(self):
        return self._name

    @property
    def epoch_max(self) -> Optional[int]:
        return getattr(self, "_epoch_max", None)

    @property
    def learning_rate(self) -> Optional[float]:
        return getattr(self, "_learning_rate", None)

    @property
    def discount(self) -> Optional[float]:
        return getattr(self, "_discount", None)

    @property
    def explo_rate(self) -> Optional[float]:
        return getattr(self, "_explo_rate", None)

    @property
    def explo_decay(self) -> Optional[float]:
        return getattr(self, "_explo_decay", None)

    @property
    def max_action(self) -> Optional[int]:
        return getattr(self, "_max_action", None)

    @property
    def reward_green_apple(self) -> Optional[FloatT]:
        return getattr(self, "_reward_green_apple", None)

    @property
    def reward_lose(self) -> Optional[FloatT]:
        return getattr(self, "_reward_lose", None)

    @property
    def reward_nothing(self) -> Optional[FloatT]:
        return getattr(self, "_reward_nothing", None)

    @property
    def reward_red_apple(self) -> Optional[FloatT]:
        return getattr(self, "_reward_red_apple", None)

    @property
    def reward_win(self) -> Optional[FloatT]:
        return getattr(self, "_reward_win", None)

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
        try:
            if os.path.exists(self.HISTORY_FILE):
                df: DataFrame = read_csv(self.HISTORY_FILE)
            else:
                df: DataFrame = DataFrame(columns=self.HISTORY_COLUMNS)
            data: Dict = {
                "model": self.name,
                "discount": self._discount,
                "exploration_rate": self._explo_rate,
                "learning_rate": self._learning_rate,
                "max_action": self._max_action,
                "epoch": self._epoch_max,
                "green_apple_reward": self._reward_green_apple,
                "win_reward": self._reward_win,
                "red_apple_malus": self._reward_red_apple,
                "nothing_malus": self._reward_nothing,
                "loose_malus": self._reward_lose,
                "num_snake_hit": sum(self._snake_hit),
                "num_wall_hit": sum(self._wall_hit),
                "snake_mean_rewards": mean(self._reward_history),
                "snake_max_rewards": max(self._reward_history),
                "snake_mean_len": mean(self._snake_len_history),
                "snake_max_len": max(self._snake_len_history)
            }
            row = Series(data).reindex(df.columns)
            df.loc[len(df)] = row
            df.to_csv('history.csv', index=False)
            logger.info("Training data historized")
        except Exception as e:
            logger.error(f"An error occured on the history updating: {e}")

    def plot_training_metrics(self,):
        create_folder(self.PLOT_FOLDER)
        create_folder(f"{self.PLOT_FOLDER}/{self.name}")
        self._plot_death()
        self._plot_apple()
        self._plot_reward()
        self._plot_snake_len()

    def _get_reward(self, idx_move: int) -> FloatT:
        reward: FloatT = self._reward_nothing
        self._action += 1
        try:
            self._env.change_move(list(Movement)[idx_move], from_ia=True)
            self._env.update()
        except SnakeLoose as exc:
            reward = self._reward_lose
            self._done = True
            if type(exc) == SnakeLooseWall: self._wall_hit[self._epoch] = 1
            elif type(exc) == SnakeLooseTail: self._snake_hit[self._epoch] = 1
        except SnakeRedApple:
            reward = self._reward_red_apple
            self._red_apple += 1
        except SnakeGreenApple:
            reward = self._reward_green_apple
            self._green_apple += 1
        except SnakeWin:
            reward = self._reward_win
            self._done = True
        self._acc_reward += reward
        if not self._done:
            self._state = self._retrieve_state_from_env()
        return reward

    def _reset_training(self):
        self._epoch = 0
        self._explo_rate_in_use = self._explo_rate
        self._reward_history = self._base.copy()
        self._snake_len_history = self._base.copy()
        self._green_apple_history = self._base.copy()
        self._red_apple_history = self._base.copy()
        self._wall_hit = self._base.copy()
        self._snake_hit = self._base.copy()

    def _reset_epoch(self):
        self._env.reset()
        self._acc_reward = 0.
        self._green_apple = 0
        self._red_apple = 0
        self._action = 0
        self._done = False
        self._state = self._retrieve_state_from_env()
        if ((self._explo_rate_in_use - self._explo_decay) > 1e-8):
            self._explo_rate_in_use -= self._explo_decay

    def _save_epoch(self):
        e: int = self._epoch
        self._reward_history[e] = self._acc_reward
        self._snake_len_history[e] = self._env.snake.size
        self._green_apple_history[e] = self._green_apple
        self._red_apple_history[e] = self._red_apple
        logger.info(f"Epoch {self._epoch} -> Reward: {round(self._acc_reward, 2)}, Snake size: {self._env.snake.size}")
        self._epoch += 1

    def _retrieve_state_from_env(self) -> List[float]:
        state_data: Dict[str, List[int]] = self._retrieve_state_data()
        # 1/distance: 1 = adjacent, 0 = nothing in this direction
        return [round(1. / d, 2) if d else 0. for v in state_data.values() for d in v]

    def _retrieve_state_data(self) -> Dict[str, List[int]]:
        snake_view: List[str] = self._env.snake_view()
        state_data: Dict[str, List[int]] = {
            "d_dist": [0, 0, 0, 0],
            "ga_dist": [0, 0, 0, 0],
            "ra_dist": [0, 0, 0, 0],
        }
        head_pos: int = self._env.snake.pos[0]
        idx: int = 1
        threated: List[int] = []
        while len(threated) < 4:
            for i in range(4):
                if i in threated:
                    continue
                if self._check_state_value(
                    state_data,
                    i,
                    snake_view[int(head_pos + idx * list(self._env.map_move.values())[i])],
                    idx,
                ) == True:
                    threated.append(i)
            idx += 1
        return state_data

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

    def _plot_death(self):
        wall_death: List[int] = self._acc_list(self._wall_hit)
        snake_death: List[int] = self._acc_list(self._snake_hit)
        df = DataFrame(prep_to_dict(
            wall_death,
            snake_death,
            "wall",
            "snake"
        ))
        fig = px.line(
            df,
            x="epoch", y="data", color="discriminant",
            labels={"data": "death"}
        )
        self._save_plot(fig, "death")
    
    def _plot_apple(self):
        green_apple: List[int] = self._green_apple_history
        red_apple: List[int] = self._red_apple_history
        df = DataFrame(prep_to_dict(
            green_apple,
            red_apple,
            "green_apple",
            "red_apple",
        ))
        fig = px.line(
            df,
            x="epoch", y="data", color="discriminant",
            labels={"data": "apple"}
        )
        self._save_plot(fig, "apple")
    
    def _plot_reward(self):
        reward: List[int] = self._reward_history
        df = DataFrame()
        df["reward"] = reward
        fig = px.line(
            df,
            x=df.index, y='reward'
        )
        self._save_plot(fig, "reward")
    
    def _plot_snake_len(self):
        snake_len: List[int] = self._snake_len_history
        df = DataFrame()
        df["snake_len"] = snake_len
        fig = px.line(
            df,
            x=df.index, y="snake_len"
        )
        self._save_plot(fig, "snake_len")
    
    def _save_plot(self, fig: Figure, plot_name: str):
        fig.write_html(f"{self.PLOT_FOLDER}/{self.name}/{plot_name}.html", auto_open=True)
    
    @staticmethod
    def _acc_list(source: List) -> List:
        acc: List = []
        if not len(source): return acc 
        acc.append(source[0])
        for i in range(1, len(source)):
            acc.append(acc[i - 1] + source[i])
        return acc
    
    @staticmethod
    def _acc_list_mean(source: List) -> List:
        acc_mean: List = []
        if not len(source): return acc_mean
        acc_mean.append(source[0])
        for i in range(1, len(source)):
            acc_mean.append(mean(source[:i]))
        return acc_mean

    @staticmethod
    def print_map(map: List[str], width:int):
        logger.info('\n')
        for i in range(len(map)):
            print(map[i],end="")
            if not (i+1) % width and i:
                print()
        print()

    @staticmethod
    def _is_key_threated(state_data: Dict[str, List[int]], idx: int):
        stop: bool = False
        for key in state_data.keys():
            if state_data[key][idx]:
                stop = True
                break
        return stop

    @staticmethod
    def create_map_action():
        new_array = ndarray(len(Movement), dtype=FloatT)
        for i in range(len(new_array)):
            new_array[i] = log(randbelow(100) + EPS)
        return new_array

    @staticmethod
    def rand_int(max: int = RAND_MAX) -> int:
        return randbelow(max)

    @staticmethod
    def _get_hashed_state(state: List[FloatT]) -> str:
        return hash_list(state)
