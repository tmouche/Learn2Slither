import tkinter

from typing import Sequence, List

from environment import Environment
from logger import logger
from exception import (
    WindowToLarge,
    UnknowCharInMap
)

class Square():

    size: int
    a_pos: list[int]

    color: str

    def __init__(self, size: int, a_pos: List[int], color: str):
        self.size = size
        self.a_pos = a_pos
        self.color = color

    def draw(self, canvas: tkinter.Canvas):
        canvas.create_rectangle(
            self.a_pos[0], self.a_pos[1],
            self.a_pos[0] + self.size, self.a_pos[1] + self.size, 
            fill=self.color
        )

class Grid(tkinter.Canvas):

    width: int
    height: int

    objects: Sequence[Square]

    CASE_SIZE = 20
    PADDING = 5

    EVEN_COLOR = "DarkKhaki"
    GREEN_APPEL_COLOR = "Chartreuse"
    ODD_COLOR = "OliveDrab"
    RED_APPLE_COLOR = "Tomato"
    SNAKE_HEAD_COLOR = "DarkBlue"
    SNAKE_BODY_COLOR = "CornflowerBlue"
    WALL_COLOR = "Black"
 
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
        self.place(x=self.PADDING, y=self.PADDING)
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
                    self.CASE_SIZE,
                    a_pos.copy(),
                    "default"
                )
            )


    def draw(self):
        i = 0
        while i < 375:
            self.objects[i].draw(self)
            i += 1


class Application(tkinter.Tk):

    width: int
    height: int

    env: Environment

    game_grid: Grid

    TITLE = "Learn2Slither"
    WHITE_SPACE = 150

    def __init__(self, game_env: Environment):

        super().__init__()
        self.title(self.TITLE)

        self.env = game_env

        self.game_grid = Grid(self, self.env.width, self.env.height)
        
        self.width: int = self.game_grid.width + self.game_grid.PADDING * 2
        self.height: int = self.game_grid.height + self.game_grid.PADDING + self.WHITE_SPACE

        if self.winfo_screenwidth() < self.width or self.winfo_screenheight() < self.height:
            raise WindowToLarge()

        self.minsize(width=self.width, height=self.height)
        self.maxsize(width=self.width, height=self.height)
        self.update_grid()
        self.game_grid.draw()
        logger.info(f"Window size: w:{self.width}, h:{self.height}")

    def update_grid(self):
        for i in range(len(self.env.map)):
            match self.env.map[i]:
                case 'G':
                    self.game_grid.objects[i].color = self.game_grid.GREEN_APPEL_COLOR 
                case 'H':
                    self.game_grid.objects[i].color = self.game_grid.SNAKE_HEAD_COLOR
                case 'O':
                    if i % 2:
                        self.game_grid.objects[i].color = self.game_grid.EVEN_COLOR
                    else:
                        self.game_grid.objects[i].color = self.game_grid.ODD_COLOR
                case 'R':
                    self.game_grid.objects[i].color = self.game_grid.RED_APPLE_COLOR
                case 'S':
                    self.game_grid.objects[i].color = self.game_grid.SNAKE_BODY_COLOR
                case 'W':
                    self.game_grid.objects[i].color = self.game_grid.WALL_COLOR
                case _:
                    raise UnknowCharInMap()


    def launch(self):
        self.mainloop()
    
