from logger import logger
from services.agent import Agent
from services.environment import Environment
from typing import Dict, Tuple
from utils.constant import EPS
from utils.dataclass import AgentSettings
from utils.enum import AgentState, Mode, Movement
from utils.types import ArrayF, FloatT

import numpy as np

from exception import (
    AgentFileInitFail,
    Format,
    SnakeGreenApple,
    SnakeLoose,
    SnakeRedApple,
    SnakeWin,
    UnexpectedException,
)
from random import randint
from utils_file import index_max, hash_list

import json

EPS = 1e-15

class AgentQL(Agent):

    q_matrix: Dict[str, ArrayF]

    STARTER: ArrayF

    def __init__(
        self, 
        env: Environment,
        agent_settings: AgentSettings
    ):
        super().__init__(env=env, agent_settings=agent_settings)
        self.STARTER = np.ndarray(4, dtype=FloatT)
        for i in range(len(self.STARTER)):
            self.STARTER[i] = 0. + EPS

        self.q_matrix = {}
        
    @classmethod
    def load(cls, path_to_agent: str):
        pass

    def train(self):
        if self.state != AgentState.CLUELESS:
            logger.info(f"Agent in state: {self.state.value}")
            return
        self.state = AgentState.IN_TRAINING
        for e in range(self.EPOCH):
            self._env.reset()
            done: bool = False
            state: str = self._get_hashed_state()
            for a in range(self.MAX_ACTION):
                idx_move: int = self.policy(Mode.TRAIN, state)
                logger.info(f"At epoch {e}:{a} action took is {list(Movement)[idx_move].name}")
                self._env.change_move(list(Movement)[idx_move], from_ia=True)
                done, reward = self._get_reward()
                next_state: str = self._update_q_matrix(state, idx_move, done, reward)
                if done == True:
                    break
                state = next_state
            if (self.EXPLO_RATE - self.EXPLO_DECAY > 1e-5):
                self.EXPLO_RATE -= self.EXPLO_DECAY
        self.state = AgentState.TRAINED
        logger.info("Training Done")

    def _get_reward(self) -> Tuple[bool, FloatT]:
        done: bool = False
        reward: FloatT = self.REWARD_NOTHING
        try:
            self._env.update()
        except SnakeLoose:
            reward = self.REWARD_LOOSE
            done = True
        except SnakeRedApple:
            reward = self.REWARD_RED_APPLE
        except SnakeGreenApple:
            reward = self.REWARD_GREEN_APPLE
        except SnakeWin:
            reward = self.REWARD_WIN
            done = True
        return (done, reward)

    def _get_hashed_state(self) -> str:
        return hash_list(self._retrieve_state_from_env())

    def _update_q_matrix(
        self,
        state: str,
        idx_move: int,
        done: bool,
        reward: FloatT
    ) -> str:
        c_state_rewards: ArrayF = self.q_matrix.get(state)
        if done == False:
            next_state: str = self._get_hashed_state()
            n_max_reward: float = max(self.q_matrix.get(next_state, [0]))
        else:
            n_max_reward = 0.
        c_state_rewards[idx_move] = (
            c_state_rewards[idx_move]
            + self.LEARNING_RATE
            * (
                reward
                + (self.DISCOUNT * n_max_reward)
                - c_state_rewards[idx_move]
            )
        )
        return next_state

    def take_action(self) -> Movement:
        state: str = hash_list(self._retrieve_state_from_env())
        idx_move:int = self.policy(Mode.TEST, state)
        return list(Movement)[idx_move]

    def policy(self, mode: Mode, state: str) -> int:
        idx_move: int
        rand: int = randint(0, self._RAND_MAX)
        if mode == Mode.TRAIN:
            if rand < self.EXPLO_RATE * self.EPOCH or not self.q_matrix.get(state):
                idx_move = rand%4
                self.q_matrix[state]= self.STARTER.copy()
            else:
                idx_move, _ = index_max(self.q_matrix.get(state))

        else:
            if self.q_matrix.get(state):
                idx_move, _ = index_max(self.q_matrix.get(state))
            else:
                logger.info("No experience found for the given state, random one choose")
                idx_move = rand%4
        return idx_move
        
