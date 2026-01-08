from collections import deque
from typing import Sequence
from random import randint
from abc import abstractmethod

class Basket:

    pos: Sequence[int]

    def __init__(self):
        self.pos = []

    def new(self, npos: int):
        self.pos.append(npos)

    def delete(self, npos: int):
        self.pos.remove(npos)

    @abstractmethod
    def collision(self, snake_pos: int) -> bool:
        if self.pos.index(snake_pos):
            return True
        return False

class Environment:
    
    width:int
    height:int

    map:Sequence[str]

    green_apples: Sequence[int]
    red_apples: Sequence[int]

    snake: Sequence[int]
    direction: str

    START_GA = 2
    START_RA = 1
    START_SNAKE = 3

    def __init__(self, w:int, h:int):
        
        self.width = w
        self.height = h

        self.map = ['O' for i in range(self.width * self.height)]
        

        self.red_apples = []
        self.green_apples = []
        self.snake = deque()

        for _ in range(self.START_SNAKE - 1):
            self.increase_snake()
        for _ in range(self.START_GA):
            self.new_green_apple()
        for _ in range(self.START_RA):
            self.new_red_apple()

        

    def new_green_apple(self):
        pass

    def new_red_apple(self):
        pass

    def increase_snake(self):
        pass

    def decrease_snake(self):
        pass