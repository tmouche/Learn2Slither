from core.agent import Agent
from core.environment import Environment
from utils.logger import logger
from pickle import dump, load
from secrets import choice
from typing import Dict, Generator, List, override, Tuple
from utils.dataclass import AgentSettings
from utils.enum import Movement
from utils.types import ArrayF, FloatT
from utils.exception import (
    AgentFileSave,
)
from utils.utils_file import index_max
import os


class AgentQL(Agent):

    q_matrix: Dict[str, ArrayF]

    def __init__(
        self, 
        env: Environment,
        agent_settings: AgentSettings,
    ):
        super().__init__(env=env, agent_settings=agent_settings)
        self.q_matrix = {}

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
        path_to_file: str = f"{self.SAVE_FOLDER}/{name}.pkl" 
        with open(path_to_file, "rb") as f:
            self.q_matrix = load(f)

    @override
    def save(self):
        path_to_file: str = f"{self.SAVE_FOLDER}/{self.name}.pkl"
        try:
            os.mkdir(self.SAVE_FOLDER)
        except FileExistsError:
            pass
        except Exception as exc:
            logger.error(f'An unexpected error is happened on creating save folder: {exc}')
            raise AgentFileSave("")
        with open(path_to_file, "wb") as f:
            dump(self.q_matrix, f)

    @override
    def take_action(self) -> Movement:
        self._state = self._retrieve_state_from_env()
        state: str = self._get_hashed_state(self._state)
        idx_move:int = self._policy(state, self.q_matrix.get(state) is None)
        return list(Movement)[idx_move]

    def _critical_loop(self):
        hashed_state: str = self._get_hashed_state(self._state)
        idx_move: int = self._define_move(hashed_state)
        reward = self._get_reward(idx_move)
        self._update_q_matrix(hashed_state, idx_move, reward)

    def _update_q_matrix(
        self,
        state: str,
        idx_move: int,
        reward: FloatT
    ):
        c_state_rewards: ArrayF = self.q_matrix.get(state)
        if self._done == False:
            next_state: str = self._get_hashed_state(self._state)
            n_max_reward: float = max(self.q_matrix.get(next_state, [0]))
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

    def _define_move(self, hashed_state: str):
        random_action: bool = self.rand_int() < (self._explo_rate_in_use * 100)
        if self.q_matrix.get(hashed_state) is None:
            self.q_matrix[hashed_state] = self.create_map_action()
            random_action = True
        idx_move: int = self._policy(hashed_state, random_action)
        return idx_move

    def _policy(self, state: str, random_action: bool = False) -> int:
        idx_move: int = 0
        if random_action == True:
            idx_move = choice(self._movement_default)
        else:
            idx_move, _ = index_max(self.q_matrix.get(state))
        return idx_move
