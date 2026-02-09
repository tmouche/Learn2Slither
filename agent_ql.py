from agent import Agent
from enumeration import Movement, Mode
from environment import Environment
from exception import (
    AgentFileInitFail,
    Format,
    SnakeGreenApple,
    SnakeLoose,
    SnakeRedApple,
    SnakeWin,
    UnexpectedException,
)
from logger import logger
from random import randint
from typing import Dict, List
from utils import index_max, hash_list

import json

EPS = 1e-15

class AgentQL(Agent):

    q_matrix: Dict[str, List[float]]

    STARTER: List[float] = [0.+EPS, 0.+EPS, 0.+EPS, 0.+EPS]

    REWARD_LOOSE: float = -5
    REWARD_RED_APPLE: float = -1
    REWARD_NOTHING: float = -.1
    REWARD_GREEN_APPLE: float = 2.
    REWARD_WIN: float = 50.

    def __init__(self, **kwargs):

        if "q_matrix_path" in kwargs.keys():
            self.__init_trained(kwargs.get("env"), kwargs.get("q_matrix_path"))
        else:
            self.__init_training(
                kwargs.get("env"),
                kwargs.get("epoch"),
                kwargs.get("l_rate"),
                kwargs.get("discount"),
                kwargs.get("e_rate"),
                kwargs.get("e_decay"),
                kwargs.get("max_action"),
                kwargs.get("path_to_save")
            )        


    def __init_trained(self, env: Environment, q_matrix_path: str):
        try:
            f = None
            f = open(q_matrix_path, "r")
            raw_data: str = f.read()
        except FileNotFoundError as fileErr:
            logger.error(f"{self.__class__.__qualname__}: {fileErr}")
            raise AgentFileInitFail(q_matrix_path)
        except Exception as e:
            logger.error(f"{self.__class__.__qualname__}: {e}")
            raise UnexpectedException()
        finally:
            if f:
                f.close()
        
        try:
            self.q_matrix = json.loads(raw_data)
        except json.JSONDecodeError as jsonErr:
            logger.error(f"{self.__class__.__qualname__}: {jsonErr}")
            raise Format("json")
        self._env = env

    def __init_training(
        self, 
        env: Environment,
        epoch:int,
        l_rate:float,
        discount:float,
        e_rate:float,
        e_decay:float,
        max_action:int,
        path_to_save: str | None = None
    ):
        super().__init__(
            env,
            epoch,
            l_rate,
            discount,
            e_rate,
            e_decay,
            max_action,
            path_to_save
        )

        self.q_matrix = None

    def train(self):
        if self.q_matrix:
            logger.info("Agent already trained")
            return
        self.q_matrix = {}
        for e in range(self.EPOCH):
            self._env.reset()
            done: bool = False
            state: str = hash_list(self._retrieve_state_from_env())
            for a in range(self.MAX_ACTION):
                idx_move: int = self.policy(Mode.TRAIN, state)
                logger.info(f"At epoch {e}:{a} action took is {list(Movement)[idx_move].name}")
                # self.print_map(self._env.snake_view(), self._env.width)
                self._env.change_move(list(Movement)[idx_move], from_ia=True)
                reward: float = -0.1
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
                c_state_rewards: List[float] = self.q_matrix.get(state)
                if done == False:
                    next_state: List[float] = hash_list(self._retrieve_state_from_env())
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
                if done == True:
                    break
                state = next_state
            if (self.EXPLO_RATE - self.EXPLO_DECAY > 1e-5):
                self.EXPLO_RATE -= self.EXPLO_DECAY
        self.export_to_json()
        logger.info("Training Done")

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
                logger.info(f"No experience found for the given state, random one choose")
                idx_move = rand%4
        return idx_move

    def export_to_json(self):
        if not self.PATH_TO_SAVE:
            return 
        raw_data: str = json.dumps(self.q_matrix, indent=4)

        try:
            f = None
            f = open(self.PATH_TO_SAVE, "w")
            f.write(raw_data)
        except FileNotFoundError as fileErr:
            logger.error(f"{self.__class__.__qualname__}: {fileErr}")
            raise AgentFileInitFail(self.PATH_TO_SAVE)
        except Exception as e:
            logger.error(f"{self.__class__.__qualname__}: {e}")
            raise UnexpectedException()
        finally:
            if f:
                f.close()
        
