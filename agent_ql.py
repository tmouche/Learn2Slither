from agent import Agent
from enumeration import Movement, Mode
from environment import Environment
from hashlib import sha1
from random import randint
from typing import Dict, List


class AgentQL(Agent):

    q_matrix: Dict[str, List[float]]


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
        super.__init__(
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
        
        for _ in range(self.EPOCH):
            self._env.reset()
            for _ in range(self.MAX_ACTION):
                pass


    def policy(self, mode: Mode, state: str):
        move: Movement
        if mode == Mode.TRAIN:
            rand: int = randint()
            if 1 / rand < self.EXPLO_RATE:
                pass
            else:
                move = list(move)[rand%4]
        else:
            move = self.q_matrix[state]
        return move
                 


    



# test = sha1("encode".encode()).hexdigest()