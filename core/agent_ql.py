from core.agent import Agent
from core.environment import Environment
from utils.logger import logger
from math import log
from numpy import mean, ndarray
from pandas import DataFrame, Series, read_csv
from pickle import dump, load
from plotly.graph_objects import Figure
from secrets import choice, randbelow
from typing import Dict, Generator, List, override, Tuple
from utils.constant import EPS, RAND_MAX
from utils.dataclass import AgentSettings
from utils.enum import Movement
from utils.types import ArrayF, FloatT
from utils.exception import (
    AgentFileSave,
    SnakeGreenApple,
    SnakeLoose,
    SnakeLooseTail,
    SnakeLooseWall,
    SnakeRedApple,
    SnakeWin,
)
from utils.utils_file import index_max, hash_list, create_folder, prep_to_dict
import os
import plotly.express as px

EPS = 1e-15

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
HISTORY_FILE = "history.csv"
SAVE_FOLDER = "models"
PLOT_FOLDER = "plots"

class AgentQL(Agent):

    q_matrix: Dict[str, ArrayF]
    
    green_apple_history: List[int]
    red_apple_history: List[int]
    reward_history: List[float]
    snake_len_history: List[int]
    snake_hit: List[int]
    wall_hit: List[int]
    
    _epoch: int
    _explo_rate_in_use: FloatT

    _acc_reward: FloatT
    _action: int
    _done: bool
    _green_apple: int
    _red_apple: int
    _state: str

    _base: List[int]
    _movement_default: List[int]

    def __init__(
        self, 
        env: Environment,
        agent_settings: AgentSettings,
    ):
        super().__init__(env=env, agent_settings=agent_settings)
        self.q_matrix = {}
        self._movement_default = range(len(Movement))
        self._base = [0 for _ in range(self._epoch_max)]

    @override
    def train(self):
        self._reset_training()
        while self._epoch < self._epoch_max:
            self._reset_epoch()
            while self._action < self._max_action and self._done == False:
                self._critical_loop()
            self._save_epoch()

    @override
    def train_sequentially(self) -> Generator:
        self._reset_training()
        while self._epoch < self._epoch_max:
            self._reset_epoch()
            while self._action < self._max_action and self._done == False:
                self._critical_loop()
                yield
            self._save_epoch()

    @override
    def load(self, name: str):
        path_to_file: str = f"{SAVE_FOLDER}/{name}.pkl" 
        with open(path_to_file, "rb") as f:
            self.q_matrix = load(f)

    @override
    def save(self):
        path_to_file: str = f"{SAVE_FOLDER}/{self.name}.pkl"
        try:
            os.mkdir(SAVE_FOLDER)
        except FileExistsError:
            pass
        except Exception as exc:
            logger.error(f'An unexpected error is happened on creating save folder: {exc}')
            raise AgentFileSave("")
        with open(path_to_file, "wb") as f:
            dump(self.q_matrix, f)

    @override
    def take_action(self) -> Movement:
        state: str = self._get_hashed_state()
        idx_move:int = self.policy(state, self.q_matrix.get(state) is None)
        return list(Movement)[idx_move]

    @override
    def plot_training_metrics(self,):
        create_folder(PLOT_FOLDER)
        create_folder(f"{PLOT_FOLDER}/{self.name}")
        self._plot_death()
        self._plot_apple()
        self._plot_reward()
        self._plot_snake_len()

    @override
    def save_training_metrics(self):
        try:
            if os.path.exists(HISTORY_FILE):
                df: DataFrame = read_csv(HISTORY_FILE)
            else:
                df: DataFrame = DataFrame(columns=HISTORY_COLUMNS)
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
                "num_snake_hit": sum(self.snake_hit),
                "num_wall_hit": sum(self.wall_hit),
                "snake_mean_rewards": mean(self.reward_history),
                "snake_max_rewards": max(self.reward_history),
                "snake_mean_len": mean(self.snake_len_history),
                "snake_max_len": max(self.snake_len_history)
            }
            row = Series(data).reindex(df.columns)
            df.loc[len(df)] = row
            df.to_csv('history.csv', index=False)
            logger.info("Training data historized")
        except Exception as e:
            logger.error(f"An error occured on the history updating: {e}")

    def _critical_loop(self):
        idx_move: int = self._define_move()
        reward = self._get_reward(idx_move)
        self._update_q_matrix(self._state, idx_move, reward)
        if ((self._explo_rate_in_use - self._explo_decay) > 1e-8):
            self._explo_rate_in_use -= self._explo_decay

    def _reset_training(self):
        self._epoch = 0
        self._explo_rate_in_use = self._explo_rate
        self.reward_history = self._base.copy()
        self.snake_len_history = self._base.copy()
        self.green_apple_history = self._base.copy()
        self.red_apple_history = self._base.copy()
        self.wall_hit = self._base.copy()
        self.snake_hit = self._base.copy()

    def _reset_epoch(self):
        self._env.reset()
        self._acc_reward = 0.
        self._green_apple = 0
        self._red_apple = 0
        self._action = 0
        self._done = False
        self._state = self._get_hashed_state()

    def _save_epoch(self):
        e: int = self._epoch
        self.reward_history[e] = self._acc_reward
        self.snake_len_history[e] = self._env.snake.size
        self.green_apple_history[e] = self._green_apple
        self.red_apple_history[e] = self._red_apple
        logger.info(f"Epoch {self._epoch} -> Reward: {round(self._acc_reward, 2)}, Snake size: {self._env.snake.size}")
        self._epoch += 1

    def _get_reward(self, idx_move: int) -> Tuple[bool, FloatT]:
        reward: FloatT = self._reward_nothing
        try:
            self._env.change_move(list(Movement)[idx_move], from_ia=True)
            self._env.update()
        except SnakeLoose as exc:
            reward = self._reward_lose
            self._done = True
            if type(exc) == SnakeLooseWall: self.wall_hit[self._epoch] = 1
            elif type(exc) == SnakeLooseTail: self.snake_hit[self._epoch] = 1
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
        return reward

    def _get_hashed_state(self):
        return hash_list(self._retrieve_state_from_env())

    def _update_q_matrix(
        self,
        state: str,
        idx_move: int,
        reward: FloatT
    ):
        c_state_rewards: ArrayF = self.q_matrix.get(state)
        if self._done == False:
            self._state = self._get_hashed_state()
            n_max_reward: float = max(self.q_matrix.get(self._state, [0]))
        else:
            n_max_reward = 0.
        c_state_rewards[idx_move] = (
            c_state_rewards[idx_move]
            + (self._learning_rate
            * (
                reward
                + (self._discount * n_max_reward)
                - c_state_rewards[idx_move]
            ))
        )

    def _define_move(self):
        random_action: bool = self.rand_int() < (self._explo_rate_in_use * 100)
        if self.q_matrix.get(self._state) is None:
            self.q_matrix[self._state] = self.create_map_action()
            random_action = False
        idx_move: int = self.policy(self._state, random_action)
        return idx_move

    def policy(self, state: str, random_action: bool = False) -> int:
        idx_move: int = 0
        if random_action == True:
            idx_move = choice(self._movement_default)
        else:
            idx_move, _ = index_max(self.q_matrix.get(state))
        return idx_move

    @staticmethod
    def create_map_action():
        new_array = ndarray(len(Movement), dtype=FloatT)
        for i in range(len(new_array)):
            new_array[i] = log(randbelow(100) + EPS)
        return new_array

    @staticmethod
    def rand_int(max: int = RAND_MAX) -> int:
        return randbelow(max)

    def _plot_death(self):
        wall_death: List[int] = self._acc_list(self.wall_hit)
        snake_death: List[int] = self._acc_list(self.snake_hit)
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
        green_apple: List[int] = self.green_apple_history
        red_apple: List[int] = self.red_apple_history
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
        reward: List[int] = self.reward_history
        df = DataFrame()
        df["reward"] = reward
        fig = px.line(
            df,
            x=df.index, y='reward'
        )
        self._save_plot(fig, "reward")

    def _plot_snake_len(self):
        snake_len: List[int] = self.snake_len_history
        df = DataFrame()
        df["snake_len"] = snake_len
        fig = px.line(
            df,
            x=df.index, y="snake_len"
        )
        self._save_plot(fig, "snake_len")

    def _save_plot(self, fig: Figure, plot_name: str):
        fig.write_html(f"{PLOT_FOLDER}/{self.name}/{plot_name}.html", auto_open=True)

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