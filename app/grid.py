from app.square import Square
from typing import List
from pygame import Surface

class Grid():
    BLACK_COLOR = "black"
    EVEN_COLOR = "DarkKhaki"
    GREEN_APPEL_COLOR = "Chartreuse"
    ODD_COLOR = "OliveDrab"
    RED_APPLE_COLOR = "Tomato"
    SNAKE_HEAD_COLOR = "DarkBlue"
    SNAKE_BODY_COLOR = "CornflowerBlue"
    WALL_COLOR = "DarkGreen"
    
    CASE_SIZE = 20

    objs: List[Square]
    
    _width: int
    _height: int

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __init__(
        self,
        env_width: int,
        env_height: int
    ):
        self._width = env_width * self.CASE_SIZE + 2
        self._height = env_height * self.CASE_SIZE + 2
        self.objs = []
        a_pos: List[int] = [0, 0]
        for i in range(env_width * env_height):
            if not i % env_width and i:
                a_pos[0] = 0
                a_pos[1] += self.CASE_SIZE
            elif i:
                a_pos[0] += self.CASE_SIZE
            self.objs.append(
                Square(
                    self.CASE_SIZE,
                    a_pos.copy(),
                    color=self.BLACK_COLOR
                )
            )

    def draw(self, surface: Surface):
        for i in range(len(self.objs)):
            self.objs[i].draw(surface)
