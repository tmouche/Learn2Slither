from agent import Agent
from enumeration import Movement, Mode
from environment import Environment
from exception import (
    SnakeGreenApple,
    SnakeLoose,
    SnakeRedApple,
    SnakeWin
)
from logger import logger
from random import randint
from typing import Dict, List
from utils import index_max, hash_list

class AgentQL(Agent):

    q_matrix: Dict[str, List[float]]

    STARTER: List[float] = [0., 0., 0., 0.]

    REWARD_LOOSE: float = -5
    REWARD_RED_APPLE: float = -.5
    REWARD_NOTHING: float = +.1
    REWARD_GREEN_APPLE: float = 1.
    REWARD_WIN: float = 2.

    def __init__(
        self, 
        env: Environment,
        epoch:int,
        l_rate:float,
        discount:float,
        e_rate:float,
        e_decay:float,
        max_action:int
    ):
        super().__init__(
            env,
            epoch,
            l_rate,
            discount,
            e_rate,
            e_decay,
            max_action
        )

        self.q_matrix = {}

    def train(self):
        explo_rate: float = self.EXPLO_RATE
        for e in range(self.EPOCH):
            self._env.reset()
            done: bool = False
            state: str = hash_list(self._retrieve_state_from_env())
            for a in range(self.MAX_ACTION):
                idx_move: int = self.policy(Mode.TRAIN, state)
                logger.info(f"At epoch {e}:{a} action took is {list(Movement)[idx_move].name}")
                self.print_map(self._env.snake_view(), self._env.width)
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
            if (explo_rate - self.EXPLO_DECAY > 1e-5):
                explo_rate -= self.EXPLO_DECAY
        logger.info("Training Done")

    def take_action(self) -> Movement:
        state: List[float] = hash_list(self._retrieve_state_from_env())
        idx_move:int = self.policy(Mode.TEST, state)
        return list(Movement)[idx_move]

    def policy(self, mode: Mode, state: str) -> int:
        idx_move: int
        rand: int = randint(0, 1000000)
        if mode == Mode.TRAIN:
            if 1 / rand < self.EXPLO_RATE and self.q_matrix.get(state):
                idx_move, _ = index_max(self.q_matrix.get(state))
            else:
                idx_move = rand%4
                self.q_matrix[state]= self.STARTER.copy()
        else:
            if self.q_matrix.get(state):
                idx_move, _ = index_max(self.q_matrix.get(state))
            else:
                logger.info(f"No experience found for the given state, random one choose")
                idx_move = rand%4
        return idx_move
                 


    



# test = sha1("encode".encode()).hexdigest()