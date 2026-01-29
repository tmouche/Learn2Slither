import tkinter

from typing import Sequence, List
from time import sleep

from environment import Environment
from logger import logger
from exception import (
    WindowToLarge,
    UnknowCharInMap
)
from enumeration import Movement

class Square():

    size: int
    a_pos: list[int]

    color: str

    ID:int

    def __init__(self, canvas: tkinter.Canvas, size: int, a_pos: List[int], color: str):
        self.size = size
        self.a_pos = a_pos
        self.color = color
        self.ID = canvas.create_rectangle(
            self.a_pos[0], self.a_pos[1],
            self.a_pos[0] + self.size, self.a_pos[1] + self.size, 
            fill=self.color
        )

    def draw(self, canvas: tkinter.Canvas):
        canvas.itemconfig(self.ID, fill=self.color)
        


class Grid(tkinter.Canvas):

    width: int
    height: int

    objects: Sequence[Square]

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

class Menu(tkinter.Canvas):

    width: int
    height: int

    human_frame: tkinter.Frame
    human_button: tkinter.Button
    ia_frame: tkinter.Frame
    ia_button: tkinter.Button
    step_frame: tkinter.Frame
    step_button: tkinter.Button
    switch_visual_frame: tkinter.Frame
    switch_visual_button: tkinter.Button
    
    BUTTON_WIDTH = 100
    BUTTON_HEIGHT = 25

    COLOR_ACTIVATE_BUTTON = "Ivory"
    COLOR_DEACTIVATE_BUTTON = "DarkGrey"



    def __init__(self, window: tkinter.Tk, menu_width: int, menu_height: int):
        self.width = menu_width
        self. height = menu_height
        super().__init__(
            window, 
            width=self.width,
            height=self.height,
            bg="ivory",
            bd=0,
            highlightthickness=0
        )
        self.pack()

        self.human_frame = None
        self.human_button = None
        self.ia_frame = None
        self.ia_button = None
        self.step_frame = None
        self.step_button = None
        self.next_step_frame = None
        self.next_step_button = None
        self.switch_visual_button = None
        self.switch_visual_frame = None

        
    def create_human_start_button(self, cmd_fnc:callable):
        self.human_frame = tkinter.Frame(
            self,
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT,
        )
        self.human_frame.pack_propagate(False)
        self.human_frame.pack()
        self.human_button = tkinter.Button(
            self.human_frame,
            text="Start Human",
            command=cmd_fnc,
            bg=self.COLOR_ACTIVATE_BUTTON
        )
        self.human_button.pack(fill="both", expand=True, side="left")
        
    def create_ia_start_button(self, cmd_fnc:callable):
        self.ia_frame = tkinter.Frame(
            self,
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT,
        )
        self.ia_frame.pack_propagate(False)
        self.ia_frame.pack()
        self.ia_button = tkinter.Button(
            self.ia_frame,
            text="Start IA",
            command=cmd_fnc,
            bg=self.COLOR_ACTIVATE_BUTTON
        )
        self.ia_button.pack(fill="both", expand=True, side="left")

    def create_switch_step_button(self, cmd_fnc: callable):
        self.step_frame = tkinter.Frame(
            self,
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT
        )
        self.step_frame.pack_propagate(False)
        self.step_frame.pack()
        self.step_button = tkinter.Button(
            self.step_frame,
            text="Switch to Step",
            command=cmd_fnc,
            bg=self.COLOR_ACTIVATE_BUTTON
        )
        self.step_button.pack(fill="both", expand=True, side="left")

    def create_next_step_button(self, cmd_fnc: callable):
        self.next_step_frame = tkinter.Frame(
            self,
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT
        )
        self.next_step_frame.pack_propagate(False)
        self.next_step_frame.pack()
        self.next_step_button = tkinter.Button(
            self.next_step_frame,
            text="Next Step",
            command=cmd_fnc,
            bg=self.COLOR_ACTIVATE_BUTTON
        )
        self.next_step_button.pack(fill="both", expand=True, side="left")

    def create_switch_visual_button(self, cmd_fnc: callable):
        self.switch_visual_frame = tkinter.Frame(
            self,
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT
        )
        self.switch_visual_frame.pack_propagate(False)
        self.switch_visual_frame.pack()
        self.switch_visual_button = tkinter.Button(
            self.switch_visual_frame,
            text="Switch visual",
            command=cmd_fnc,
            bg=self.COLOR_ACTIVATE_BUTTON
        )
        self.switch_visual_button.pack(fill="both", expand=True, side="left")

    def deactivate_start_buttons(self):
        self.human_button.configure(
            command=self.do_nothing,
            bg=self.COLOR_DEACTIVATE_BUTTON,
            activebackground=self.COLOR_DEACTIVATE_BUTTON
        )
        self.ia_button.configure(
            command=self.do_nothing,
            bg=self.COLOR_DEACTIVATE_BUTTON,
            activebackground=self.COLOR_DEACTIVATE_BUTTON
        )

    def do_nothing(self):
        pass

class Application(tkinter.Tk):

    width: int
    height: int

    env: Environment

    game_grid: Grid
    menu: Menu

    step_mode: bool
    visual_mode: bool

    TITLE = "Learn2Slither"

    GAME_SPEED_VISUAL = 200
    GAME_SPEED_NO_VISUAL = 1

    GRID_T_PADDING = 5
    GRID_L_PADDING = 5
    MENU_SIZE = 150
    MENU_T_PADDING = 10
    MENU_L_PADDING = 5
    WHITE_SPACE = MENU_SIZE + MENU_T_PADDING * 2

    def __init__(self, game_env: Environment):

        super().__init__()
        self.title(self.TITLE)

        self.env = game_env

        self.step_mode = False
        self.visual_mode = True

        self.game_grid = Grid(self, self.env.width, self.env.height)
        self.game_grid.place(x=self.GRID_L_PADDING, y=self.GRID_T_PADDING)
        
        self.width: int = self.game_grid.width + self.GRID_L_PADDING * 2
        self.height: int = self.game_grid.height + self.GRID_T_PADDING + self.WHITE_SPACE

        if self.winfo_screenwidth() < self.width or self.winfo_screenheight() < self.height:
            raise WindowToLarge()

        self.minsize(width=self.width, height=self.height)
        self.maxsize(width=self.width, height=self.height)
        self.update_grid()
        self.game_grid.draw()   

        self.menu = Menu(self, self.game_grid.width, self.MENU_SIZE)
        MENU_Y_START:int = self.GRID_L_PADDING + self.game_grid.height + self.MENU_T_PADDING
        self.menu.place(x=self.MENU_L_PADDING, y=MENU_Y_START)
        self.menu.create_human_start_button(self.game_human_loop)
        self.menu.create_ia_start_button(self.game_ia_loop)
        self.menu.create_switch_step_button(self.switch_step)
        self.menu.create_next_step_button(self.update)
        self.menu.create_switch_visual_button(self.switch_visual)

        self.bind('<Key>', self.key_handler)

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

    def key_handler(self, event: tkinter.Event):
        pressed_key: str = event.keysym
        
        if pressed_key == "Up":
            self.env.change_move(Movement.UP)
        elif pressed_key == "Down":
            self.env.change_move(Movement.DOWN)
        elif pressed_key == "Left":
            self.env.change_move(Movement.LEFT)
        elif pressed_key == "Right":
            self.env.change_move(Movement.RIGHT)

    def switch_step(self):
        self.step_mode = True if self.step_mode == False else False

    def switch_visual(self):
        self.visual_mode = True if self.step_mode == False else False

    def update(self):
        try:
            self.env.update()
        except Exception as exc:
            logger.info(f"Game Finished: {exc}")
            self.destroy()
            return 
        self.update_grid()
        self._print_map(self.env.snake_view(), self.env.width)
        logger.info(self.env.actual_move.name)
        if self.visual_mode:
            self.game_grid.draw()

    def game_ia_loop(self):
        self.menu.deactivate_start_buttons()
        if self.step_mode == False:
            # action = self.agent.take_action()
            # self.env.change_move(action)
            self.update()
        refresh_rate:int = self.GAME_SPEED_VISUAL if self.visual_mode else self.GAME_SPEED_NO_VISUAL
        self.game_grid.after(refresh_rate, self.game_human_loop)

    def game_human_loop(self):
        self.menu.deactivate_start_buttons()
        if self.step_mode == False:
            self.update()
        refresh_rate:int = self.GAME_SPEED_VISUAL if self.visual_mode else self.GAME_SPEED_NO_VISUAL
        self.game_grid.after(refresh_rate, self.game_human_loop)

    def launch(self):
        self.mainloop()

    def _print_map(self, map: Sequence[str], width:int):
        logger.info('\n')
        for i in range(len(map)):
            print(map[i],end="")
            if not (i+1) % width and i:
                print()
        print()


    
