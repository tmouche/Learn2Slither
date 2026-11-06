from core.agent import Agent
from dataclasses import dataclass
from mlp.core.model import Network
from mlp.core.layer import Layer
from mlp.ml_tools.optimizers import GradientDescent
from mlp.utils.types import ArrayF, FloatT
from pickle import dump, load
from random import SystemRandom
from secrets import choice
from typing import Dict, Generator, List, Optional, override, Tuple
from utils.constant import EPS
from utils.enum import Movement
from utils.exception import (
    AgentFileSave,
)
from utils.logger import logger
from utils.utils_file import index_max
import os
import numpy as np

TD_CLIP: FloatT = 1.

@dataclass
class Experience:
    state: List[FloatT]
    next_state: List[FloatT]
    action: Movement
    reward: FloatT
    done: bool
    label: Optional[List[FloatT]] = None
    td_error: Optional[FloatT] = 0.

class ReplayBuffer:
    _size: int
    _buffer: List[Experience]
    _priorities: List[FloatT]

    _generator: np.random.Generator

    @property
    def size(self) -> int:
        return self._size

    def __init__(self, size: int):
        self._size = size
        self._buffer = []
        self._generator = np.random.Generator(np.random.PCG64(42))

    def add(self, experience: Experience):
        if len(self._buffer) >= self._size:
            self._buffer.pop(0)
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        if batch_size < 0:
            raise ValueError("batch_size has to be > to 0")
        real_size: int = batch_size if batch_size <= len(self._buffer) else len(self._buffer)
        priorities = np.array([xp.td_error for xp in self._buffer]) ** 0.6
        probabilities: List[FloatT] = priorities / np.sum(priorities)
        sample_idx: List[int] = self._generator.choice(
            range(len(self._buffer)), size=real_size, p=probabilities
        )
        xp_sample: List[Experience] = [self._buffer[i] for i in sample_idx]
        return xp_sample
        
class AgentDQL(Agent):

    _exp_replay: ReplayBuffer
    _q_network: Network
    _t_network: Network    
    _batch_size: int
    _optimizer: GradientDescent

    def __init__(self, env, agent_settings, batch_size: int):
        super().__init__(env, agent_settings)
        self._exp_replay = ReplayBuffer(6000)
        self._batch_size = batch_size
        self._q_network = Network(
            layers=[
                Layer(shape=12),
                Layer(shape=16, activation="LeakyReLu", initializer="he_normal"),
                Layer(shape=16, activation="LeakyReLu", initializer="he_normal"),
                Layer(shape=4, activation="Nothing", initializer="xavier_normal")
            ],
            learning_rate=self.learning_rate if self.learning_rate is not None else 1,
            batch_size=self._batch_size,
            loss_function="mean_square_error"
        )
        self._optimizer = GradientDescent(
            network=self._q_network
        )
        self._t_network = self._q_network.copy()
        
    @override
    def train(self):
        self._reset_training()
        while self._epoch < self._epoch_max:
            self._reset_epoch()
            while self._action < self._max_action and self._done == False:
                self._critical_loop()
                if not self._action % 15 or self._done == True:
                    self._t_network = self._q_network.copy()
            logger.info(f"Finished at {self._action}..")
            if self._action >= self._max_action:
                logger.info("Max action reached..")
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
        path_to_file_weight: str = f"{self.SAVE_FOLDER}/{name}_weight.pkl"
        path_to_file_biaises: str = f"{self.SAVE_FOLDER}/{name}_biaises.pkl"
        with open(path_to_file_weight, "rb") as f:
            self._q_network.weights = load(f)
        with open(path_to_file_biaises, "rb") as f:
            self._q_network.biaises = load(f)

    @override
    def save(self):
        path_to_file_weight: str = f"{self.SAVE_FOLDER}/{self.name}_weight.pkl"
        path_to_file_biaises: str = f"{self.SAVE_FOLDER}/{self.name}_biaises.pkl"
        try:
            os.mkdir(self.SAVE_FOLDER)
        except FileExistsError:
            pass
        except Exception as exc:
            logger.error(f'An unexpected error is happened on creating save folder: {exc}')
            raise AgentFileSave("")
        with open(path_to_file_weight, "wb") as f:
            dump(self._q_network.weights, f)
        with open(path_to_file_biaises, "wb") as f:
            dump(self._q_network.biaises, f)

    @override
    def take_action(self) -> Movement:
        self._state = self._retrieve_state_from_env()
        idx_move:int = self._policy(self._state)
        return list(Movement)[idx_move]

    @override
    def _reset_epoch(self):
        if self._acc_reward > 0.:
            self._t_network = self._q_network.copy()
        super()._reset_epoch()

    def _critical_loop(self):
        act_state: List[FloatT] = self._state.copy()
        idx_move: int = self._define_move()
        reward: FloatT = self._get_reward(idx_move)
        self._store_experience(
            act_state,
            self._state,
            idx_move,
            reward
        )
        self._update_network()

    def _define_move(self):
        random_action: bool = self.rand_int(100) < (self._explo_rate_in_use * 100)
        idx_move: int = self._policy(random_action)
        return idx_move

    def _policy(
        self,
        random_action: bool = False
    ) -> int:
        idx_move: int = 0
        if random_action == True:
            idx_move = choice(self._movement_default)
        else:
            idx_move, _ = index_max(self._q_network.fire.full(
                    input=self._state,
                    weights=self._q_network.weights,
                    biaises=self._q_network.biaises
                )
            )
        return idx_move

    def _store_experience(
        self,
        state: List[FloatT],
        next_state: List[FloatT],
        idx_move: int,
        reward: FloatT,
    ):
        xp = self._create_experience(
            state=state,
            next_state=next_state,
            action=self._movement_default[idx_move],
            reward=reward,
            done=self._done
        )
        self._exp_replay.add(xp)

    def _update_network(
        self,
    ):
        dataset: List[Dict[str, List[FloatT]]] = self._batch_experience()
        if dataset:
            self._optimizer.full(dataset)
        
    def _batch_experience(
        self,    
    ) -> List[Dict[str, List[FloatT]]]:
        experiences_batch: List[Dict[str, List[FloatT]]] = []
        for xp in self._exp_replay.sample(self._batch_size):
            data_dict: Dict[str, List[FloatT]] = {}
            self._update_experience_inplace(xp)
            data_dict["data"] = xp.state
            data_dict["label"] = xp.label
            experiences_batch.append(data_dict)
        return  experiences_batch

    def _create_experience(
        self,
        state: List[FloatT],
        next_state: List[FloatT],
        action: int,
        done: bool,
        reward: FloatT
    ) -> Experience:
        xp = Experience(
            state=state, next_state=next_state,
            action=action, reward=reward,
            done=done
        )
        self._update_experience_inplace(xp)
        return xp

    def _update_experience_inplace(
        self,
        experience: Experience
    ):
        out_qnet: List[FloatT] = self._q_network.fire.full(
            input=experience.state,
            weights=self._q_network.weights,
            biaises=self._q_network.biaises
        )
        out_tnet: List[FloatT] = self._t_network.fire.full(
            input=experience.next_state, 
            weights=self._t_network.weights,
            biaises=self._t_network.biaises
        )
        td_target: FloatT = experience.reward
        if not experience.done:
            td_target += (self.discount * max(out_tnet))
        td_diff: FloatT = td_target - out_qnet[experience.action]
        experience.td_error = min(abs(td_diff), TD_CLIP) + EPS
        # Huber-like: the MSE gradient (y - label) is bounded to [-TD_CLIP, TD_CLIP]
        out_qnet[experience.action] += float(np.clip(td_diff, -TD_CLIP, TD_CLIP))
        experience.label = out_qnet