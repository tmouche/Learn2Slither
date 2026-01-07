import tkinter

from environment import Environment
from logger import logger
from exception import (
    MapToLarge,
    MapToSmall
)



class Application(tkinter.Tk):
    width:int
    height:int

    env:Environment

    case_size:int
    CASE_SIZE_MIN = 10
    CASE_SIZE_MAX = 250

    WIDTH_MAX = 1920
    HEIGHT_MAX = 1080 

    def __init__(self, game_env:Environment):

        self.width = 0
        self.height = 0

        self.case_size = 0

        tkinter.Tk.__init__(self)
        self.title("Learn2Slither")

        self.env = game_env
        self._define_window_size() 

        self.minsize(self.width, self.height)

        self.create_widgets()

    def _define_window_size(self):

        if self.env.width < 2 or self.env.height < 2:
            raise MapToSmall()
        if self.env.width * self.CASE_SIZE_MIN > self.WIDTH_MAX or self.env.height * self.CASE_SIZE_MIN > self.HEIGHT_MAX:
            raise MapToLarge()
        self.case_size =
        window_width = self.CASE_SIZE_MIN * self.env.width
        
        
    def create_widgets():
        pass