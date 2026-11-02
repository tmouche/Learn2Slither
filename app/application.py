from app.menu import Menu
from app.grid import Grid
from core.agent import Agent
from core.environment import Environment
from time import sleep
from typing import Generator, Sequence, Optional
from utils.logger import logger
from utils.exception import (
    WindowToLarge,
    UnknowCharInMap,
    SnakeLoose,
    SnakeWin,
    SnakeGreenApple,
    SnakeRedApple
)
from utils.enum import (
    AppStatus,
    Mode,
    Movement,
)
import os
import tkinter

TITLE_DEFAULT = "Learn2Slither"
GAME_SPEED_DEFAULT = 200
GAME_SPEED_FACTOR = 5
GAME_SPEED_MAX = 1
GAME_SPEED_MIN = 2000

class Application(tkinter.Tk):

    width: int
    height: int

    env: Environment
    agent: Agent

    game_grid: Grid
    menu: Menu

    step_mode: bool
    visual_mode: bool

    _actual_mode: Mode
    _training: Generator | None
    _status: AppStatus
    _step_mode: bool
    _started: bool
    
    GAME_SPEED_NO_VISUAL = 1

    GRID_T_PADDING = 5
    GRID_L_PADDING = 5
    MENU_SIZE = 150
    MENU_T_PADDING = 10
    MENU_L_PADDING = 5
    WHITE_SPACE = MENU_SIZE + MENU_T_PADDING * 2

    def __init__(
        self,
        game_env: Environment,
        agent: Optional[Agent] = None,
        mode: Mode = Mode.HUMAIN_PLAY,
    ):
        super().__init__()
        self.title(TITLE_DEFAULT)
        self.env = game_env
        self.agent = agent
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
        self._actual_mode = mode
        self._training = None
        self._status = AppStatus.STOP
        self._game_speed = GAME_SPEED_DEFAULT
        self._step_mode = False
        self.bind('<Key>', self.key_handler)

    def launch(self):
        self.mainloop()

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
        if self._key_handler_general(pressed_key): return
        if self._actual_mode == Mode.HUMAIN_PLAY:
           if self._key_handler_humain_play(pressed_key): return
        if self._actual_mode == Mode.AI_PLAY or self._actual_mode == Mode.AI_TRAIN:
            if self._key_handler_ai_play(pressed_key): return
        
    def _key_handler_general(self, pressed_key: str) -> bool:
        match pressed_key:
            case "Escape": self.close_app()
            case "space": self.toggle_app_status()
            case "p": self.increase_game_speed()
            case "m": self.decrease_game_speed()
            case "r": self.reset_game_speed()
            case _: return False
        return True

    def _key_handler_humain_play(self, pressed_key: str) -> bool:
        match pressed_key:
           case "Up": self.env.change_move(Movement.UP)
           case "Down": self.env.change_move(Movement.DOWN)
           case "Left": self.env.change_move(Movement.LEFT)
           case "Right": self.env.change_move(Movement.RIGHT)
           case _: return False
        return True

    def _key_handler_ai_play(self, pressed_key: str) -> bool:
        match pressed_key:
            case "Shift_L": print('PASSAGE EN STEP BY STEP / PASSAGE EN CONTINUE')
            case _: return False
        return True

    def close_app(self):
        self._status = AppStatus.STOP
        self.destroy()

    def toggle_app_status(self):
        if (self._status == AppStatus.PLAY):
            self._status = AppStatus.STOP
        else:
            self._status = AppStatus.PLAY
            match self._actual_mode:
                case Mode.HUMAIN_PLAY: self.human_play()
                case Mode.AI_TRAIN: self.ai_train()
                case Mode.AI_PLAY: self.ai_play()

    def increase_game_speed(self):
        if self._game_speed - GAME_SPEED_FACTOR < GAME_SPEED_MAX:
            logger.warning("Cannot increase the speed game: Max reached.")
            return
        self._game_speed -= GAME_SPEED_FACTOR

    def decrease_game_speed(self):
        if self._game_speed + GAME_SPEED_FACTOR > GAME_SPEED_MIN:
            logger.warning("Cannot decrease the speed game: Min reached.")
            return
        self._game_speed += GAME_SPEED_FACTOR

    def reset_game_speed(self):
        self._game_speed = GAME_SPEED_DEFAULT

    def toggle_step_mode(self):
        self._step_mode = False if self._step_mode else True

    def ai_train(self):
        if self.agent is None:
            logger.error("You need to pass an agent as argument if you want to train it through the app.")
            os._exit(1)
        if self._training is None:
            self._training = self.agent.train_sequentially()
        self._ai_train_step()

    def _ai_train_step(self):
        if self._status == AppStatus.STOP:
            return
        try:
            next(self._training)
        except StopIteration:
            self._training = None
            logger.info("Training finished.")
            return
        self.update_grid()
        self._print_map(self.env.snake_view(), self.env.width)
        logger.info(f"Movement: {self.env.actual_move.name}")
        self.game_grid.draw()
        if self._step_mode:
            self._status = AppStatus.STOP
        self.game_grid.after(self._game_speed, self._ai_train_step)

    def ai_play(self):
        if self.agent is None:
            logger.error("You need to pass an agent as argument if you want see it play.")
            os._exit(1)
        self._ai_play_step()

    def _ai_play_step(self):
        if self._status == AppStatus.STOP:
            return
        action: Movement = self.agent.take_action()
        self.env.change_move(action)
        try:
            self._update()
        except (SnakeWin, SnakeLoose):
            self.env.reset()
            return
        if self._step_mode:
            self._status = AppStatus.STOP
        self.game_grid.after(self._game_speed, self._ai_play_step)

    def human_play(self):
        self._human_play_step()

    def _human_play_step(self):
        if self._status == AppStatus.STOP:
            return
        try:
            self._update()
        except (SnakeWin, SnakeLoose):
            self.destroy()
        refresh_rate:int = self._game_speed
        self.game_grid.after(refresh_rate, self._human_play_step)

    def _update(self):
        try:
            self.env.update()
        except (SnakeLoose, SnakeWin) as exc:
            logger.info(f"Game Finished: {exc.message}, snake size: {self.env.snake.size}")
            raise exc
        except (SnakeGreenApple, SnakeRedApple):
            pass
        self.update_grid()
        self._print_map(self.env.snake_view(), self.env.width)
        logger.info(self.env.actual_move.name)
        self.game_grid.draw()

    def _print_map(self, map: Sequence[str], width:int):
        logger.info('\n')
        for i in range(len(map)):
            print(map[i],end="")
            if not (i+1) % width and i:
                print()
        print()



