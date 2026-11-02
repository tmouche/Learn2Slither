from app.square import Square
from typing import List
import tkinter

class Grid(tkinter.Canvas):

    width: int
    height: int

    objects: List[Square]

    CASE_SIZE = 20

    EVEN_COLOR = "DarkKhaki"
    GREEN_APPEL_COLOR = "Chartreuse"
    ODD_COLOR = "OliveDrab"
    RED_APPLE_COLOR = "Tomato"
    SNAKE_HEAD_COLOR = "DarkBlue"
    SNAKE_BODY_COLOR = "CornflowerBlue"
    WALL_COLOR = "DarkGreen"

    def __init__(self, window: tkinter.Tk, env_width: int, env_height: int):
        self.width = env_width * self.CASE_SIZE + 2
        self. height = env_height * self.CASE_SIZE + 2
        super().__init__(
            window,
            width=self.width,
            height=self.height,
            bg="ivory",
            bd=0,
            highlightthickness=0
        )
        self.pack()
        a_pos: List[int] = [0, 0]
        self.objects = []
        for i in range(env_width * env_height):
            if not i % env_width and i:
                a_pos[0] = 0
                a_pos[1] += self.CASE_SIZE
            elif i:
                a_pos[0] += self.CASE_SIZE

            self.objects.append(
                Square(
                    self,
                    self.CASE_SIZE,
                    a_pos.copy(),
                    "black"
                )
            )

    def draw(self):
        for i in range(len(self.objects)):
            self.objects[i].draw(self)
