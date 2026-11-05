from core.agent import Agent
from dataclasses import dataclass
from mlp.core.model import Network
from mlp.core.layer import Layer
from mlp.ml_tools.optimizers import GradientDescent
from mlp.utils.types import ArrayF, FloatT
from pickle import dump, load
from random import SystemRandom
from secrets import choice
from typing import Dict, Generator, List, override
from utils.enum import Movement
from utils.exception import (
    AgentFileSave,
)
from utils.logger import logger
from utils.utils_file import index_max
import os

@dataclass
class Experience:
    state: List[FloatT]
    next_state: List[FloatT]
    action: Movement
    reward: FloatT
    done: bool

class AgentDQL(Agent):

    _exp_replay: Dict[str, Experience]
    # _exp_replay: List[Experience]
    _q_network: Network
    _t_network: Network    
    _batch_size: int
    _optimizer: GradientDescent

    def __init__(self, env, agent_settings, batch_size: int):
        super().__init__(env, agent_settings)
        self._exp_replay = {}
        self._batch_size = batch_size
        self._q_network = Network(
            layers=[
                Layer(shape=12),
                Layer(shape=6, activation="Sigmoid", initializer="he_normal"),
                Layer(shape=4, activation="Sigmoid", initializer="he_normal")
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
        if ((self._explo_rate_in_use - self._explo_decay) > 1e-8):
            self._explo_rate_in_use -= self._explo_decay
        if reward > 0:
            self._t_network = self._q_network.copy()

    def _define_move(self):
        random_action: bool = self.rand_int() < (self._explo_rate_in_use * 100)
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
        new_experience = Experience(
            state=state,
            next_state=next_state,
            action=self._movement_default[idx_move],
            reward=reward,
            done=self._done
        )
        self._exp_replay[self._get_hashed_state(state)] = new_experience
        # if len(self._exp_replay) > 1500:
        #     self._exp_replay[self.rand_int(1499)] = new_experience
        # else:
        #     self._exp_replay.append(new_experience)
    
    def _update_network(
        self,
    ):
        dataset: List[Dict[str, List[FloatT]]] = self._batch_experience()
        self._optimizer.full(dataset)
        
    def _batch_experience(
        self,    
    ) -> List[Dict[str, List[FloatT]]]:
        index_max: int = len(self._exp_replay.values()) - 1
        # index_max: int = len(self._exp_replay) - 1
        index_to_pick: List[int] = [
            SystemRandom().randint(0, index_max) for _ in range(self._batch_size)
        ]
        experiences_batch: List[Dict[str, List[FloatT]]] = []
        for idx in index_to_pick:
            xp: Experience = list(self._exp_replay.values())[idx]
            # xp: Experience = self._exp_replay[idx]
            data_dict: Dict[str, List[FloatT]] = {}
            data_dict["data"] = xp.state
            data_dict["label"] = self._create_label(xp)
            experiences_batch.append(data_dict)
        return  experiences_batch

    def _create_label(
        self,
        xp: Experience
    ):
        out_qnet: List[FloatT] = self._q_network.fire.full(
            input=xp.state,
            weights=self._q_network.weights,
            biaises=self._q_network.biaises
        )
        # print(out_qnet)
        out_tnet: List[FloatT] = self._t_network.fire.full(
            input=xp.next_state,
            weights=self._t_network.weights,
            biaises=self._t_network.biaises
        )
        max_reward_next: FloatT = max(out_tnet)
        out_qnet[xp.action] = xp.reward
        if not xp.done:
            out_qnet[xp.action] += (self.discount * max_reward_next)
        return out_qnet