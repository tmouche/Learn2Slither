from collections import deque
from itertools import islice
from typing import Sequence, List, Dict
from random import randint
from abc import abstractmethod

from logger import logger

from enumeration import Movement

from exception import (
    MapToLarge,
    MapToSmall,
    SnakeLoose,
    SnakeWin
)


class Snake:

    pos: deque[int]

    size: int


    def __init__(self, start_pos: int, start_size: int):
        self.pos = deque()
        for i in range(start_size):
            self.pos.append(start_pos)
        self.size = start_size

    def increase(self):
        self.pos.append(self.pos[-1])
        self.size += 1

    def decrease(self):
        self.pos.pop()
        self.size -= 1

    def collision(self, check_pos: int, exclude_head: bool = False) -> bool:
        reference: deque[int] = self.pos
        if exclude_head == True:
            reference = list(islice(reference, 1, len(reference)))
        
        if check_pos in reference:
            return True
        return False

    def moove(self, next_head_pos: int):
        self.pos.appendleft(next_head_pos)
        self.pos.pop()



class Basket:

    pos: List[int]

    def __init__(self):
        self.pos = []

    def new(self, npos: int):
        self.pos.append(npos)

    def delete(self, npos: int):
        self.pos.remove(npos)

    def collision(self, check_pos: int) -> bool:
        if check_pos in self.pos:
            return True
        return False


class Environment:
    
    width:int
    height:int
    s_width: int
    s_height: int

    map: List[str]
    snake_vision_map: List[str]

    green_apple: Basket
    red_apple: Basket

    snake: Snake
    direction: str

    actual_move: Movement
    map_move: Dict[Movement, int] 
    move_counted: bool

    game_ground_size: int

    step: int

    START_GA = 2
    START_RA = 1
    START_SNAKE = 3

    START_MOVE = Movement.DOWN

    MAX_WIDTH = 1000
    MAX_HEIGHT = 1000

    def __init__(self, w:int, h:int):
        
        if w > self.MAX_WIDTH or h > self.MAX_HEIGHT:
            raise MapToLarge()
        elif w <= 5 or h <= 5:
            raise MapToSmall()

        self.width = w
        self.height = h
        self.s_width = w + 2
        self.s_height = h + 2

        self.map = ['O' for _ in range(self.width * self.height)]
        self.__init_map_move()
        self.game_ground_size = len(self.map) - (2*self.width) - (2*self.height) - 4

        self.green_apple = Basket()
        self.red_apple = Basket()
        self.snake = Snake(start_pos=self._calc_snake_first_pos(w, h), start_size=self.START_SNAKE)

        self._update_snake_map()
        self.actual_move = self.START_MOVE
        self.move_counted = True

        self.step = 0

        for _ in range(self.START_GA):
            self.new_green_apple()
        for _ in range(self.START_RA):
            self.new_red_apple()
        self._update_map()

    def __init_map_move(self):
        self.map_move = {}
        self.map_move[Movement.UP] = -self.width
        self.map_move[Movement.DOWN] = self.width
        self.map_move[Movement.LEFT] = -1
        self.map_move[Movement.RIGHT] = 1

    def update(self):
        self.step += 1
        self.move()
        self.move_counted = True
        self.check_snake_collision()
        self._update_map()
        self._update_snake_map()

    def _update_map(self):
        for i in range(len(self.map)):
            if i in self.green_apple.pos:
                self.map[i] = 'G'
            elif i in self.red_apple.pos:
                self.map[i] = 'R'
            elif i in self.snake.pos:
                if i == self.snake.pos[0]:
                    self.map[i] = 'H'
                else:
                    self.map[i] = 'S'
            elif self.is_wall(i):
                self.map[i] = 'W'
            else:
                self.map[i] = 'O'

    def _update_snake_map(self):
        s_size: int = self.s_width * self.s_height

        self.snake_vision_map = ['x' for _ in range(s_size)]
        snake_head_pos: int = self._calc_snake_first_pos(self.s_width, self.s_height)
        self.snake_vision_map[snake_head_pos] = 'H'
        left = right = up = down = snake_head_pos
        while True:
            count: List[int] = [0]
            left = self._check_pos_snake_map(left, -1, count)
            right = self._check_pos_snake_map(right, 1, count)
            up = self._check_pos_snake_map(up, -self.s_width, count)
            down = self._check_pos_snake_map(down, self.s_width, count)
            if not sum(count):
                break
                
    def _check_pos_snake_map(self, last_pos: int, move: int, count: List[int]) -> int:
        if self.snake_vision_map[last_pos] != 'W':
            temp_pos: int = last_pos + move
            if self.is_wall(temp_pos) == True:
                self.snake_vision_map[temp_pos] = 'W'
            else:
                self.snake_vision_map[temp_pos] = self.map[temp_pos]
            last_pos = temp_pos
            count.append(1)
        return last_pos

    def move(self):
        coef_move: int = self.map_move[self.actual_move]
        next_pos:int = self.snake.pos[0] + coef_move
        self.snake.moove(next_pos)

    def change_move(self, next_move: Movement):
        if (self.map_move[self.actual_move] + self.map_move[next_move] 
            and self.move_counted == True):
            self.actual_move = next_move
            self.move_counted = False

    def check_snake_collision(self):
        pos_to_check: int = self.snake.pos[0]
        if self.green_apple.collision(pos_to_check) is True:
            self.snake.increase()
            if self.snake.size == self.game_ground_size:
                raise SnakeWin()
            self.green_apple.delete(pos_to_check)
            self.new_green_apple()
        elif self.red_apple.collision(pos_to_check) is True:
            self.snake.decrease()
            if not self.snake.size:
                logger.debug("Red Apple")
                raise SnakeLoose()
            self.red_apple.delete(pos_to_check)
            self.new_red_apple()
        elif self.is_wall(pos_to_check) is True:
            logger.debug("Wall")
            raise SnakeLoose()
        elif self.snake.collision(pos_to_check, exclude_head=True) is True and self.step > self.START_SNAKE:
            logger.debug("Snake")
            raise SnakeLoose()

    def check_collision(self, next_pos:int) -> bool:
        if (
            self.green_apple.collision(next_pos) is True or
            self.red_apple.collision(next_pos) is True or
            self.is_wall(next_pos) is True or
            self.snake.collision(next_pos) is True
        ):
            return True
        return False

    def is_wall(self, pos_to_check: int) -> bool:
        if (
            pos_to_check / self.width < 1. or
            pos_to_check / self.width > self.height - 1 or
            not pos_to_check % self.width or
            pos_to_check % self.width == self.width - 1
        ):
            return True
        return False

    def new_green_apple(self):
        next_pos: int = randint(0, len(self.map))
        while self.check_collision(next_pos) == True:
            next_pos: int = randint(0, len(self.map))
        self.green_apple.pos.append(next_pos)

    def new_red_apple(self):
        next_pos: int = randint(0, len(self.map))
        while self.check_collision(next_pos) == True:
            next_pos: int = randint(0, len(self.map))
        self.red_apple.pos.append(next_pos)

    def snake_view(self) -> List[str]:
        return self.snake_vision_map
    
    def _calc_snake_first_pos(self, weight:int, height:int) -> int:
        return int(weight*(height-1)+weight/2)
    