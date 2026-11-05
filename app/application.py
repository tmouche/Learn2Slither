from app.grid import Grid
from core.environment import Environment
from core.agent import Agent
from pygame import Surface
from pygame.time import Clock
from typing import (
    Dict,
    Generator,
    List,
    Optional,
)
from utils.enum import (
    AppStatus,
    Mode,
    Movement,
)
from utils.exception import (
    WindowToLarge,
    UnknowCharInMap,
    SnakeLoose,
    SnakeWin,
    SnakeGreenApple,
    SnakeRedApple
)
from utils.logger import logger
import os
import pygame

TITLE_DEFAULT = "Learn2Slither"
GAME_SPEED_DEFAULT = 3
GAME_SPEED_FACTOR = 1
GAME_SPEED_MAX = 1000
GAME_SPEED_MIN = 1
SHIFT_L = 1073742053

class Application():
    GRID_T_PADDING = 5
    GRID_L_PADDING = 5

    _width: int
    _height: int
    _env: Environment
    _agent: Agent
    _grid: Grid
    _actual_mode: Mode
    _surface: Surface
    _status: bool
    _game_speed: int
    _training: Optional[Generator]
    _session: int

    def __init__(
        self,
        game_env: Environment,
        agent: Optional[Agent] = None,
        mode: Mode = Mode.HUMAIN_PLAY,
        title: str = TITLE_DEFAULT,
        session: int = 1,
    ):
        pygame.init()
        pygame.display.set_caption(
            title=title
        )
        self._env = game_env
        self._agent = agent
        self._grid = Grid(
            env_width=self._env.width,
            env_height=self._env.height,
        )
        self._width = self._grid.width
        self._height = self._grid.height
        self._surface = pygame.display.set_mode((self._width, self._height))
        self._status = AppStatus.STOP
        self._training = None
        self._session = 1 if session <= 0 else session
        if mode == Mode.AI_TRAIN:
            self.__init_mode_ai_train()
        elif mode == Mode.AI_PLAY:
            self.__init_mode_ai_play()
        else:
            self._actual_mode = Mode.HUMAIN_PLAY
        self._game_speed = GAME_SPEED_DEFAULT
        self._update_grid()

    def __init_mode_ai_train(self):
        self._actual_mode = Mode.AI_TRAIN
        if self._agent is None:
            logger.error("You need to pass an agent as argument if you want to train it through the app.")
            os._exit(1)
        if self._training is None:
            self._training = self._agent.train_sequentially()

    def __init_mode_ai_play(self):
        self._actual_mode = Mode.AI_PLAY
        if self._agent is None:
            logger.error("You need to pass an agent as argument if you want see it play.")
            os._exit(1)

    def launch(self, session: int = 1):
        self._status = AppStatus.PLAY
        clock = Clock()
        try:
            for _ in range(session if session >= 1 else 1):
                if self._start_game(clock) == -1:
                    break
                self._env.reset()
        except Exception as exc:
            print(exc)
        self.close_app()

    def _start_game(self, clock: Clock) -> int:
        self._status = AppStatus.PLAY
        while self._status != AppStatus.STOP:
            if self._event_handler() == -1:
                return -1
            if self._status == AppStatus.PAUSE:
                continue
            if self._actual_mode == Mode.AI_TRAIN:
                self._ai_train_step()
            elif self._actual_mode == Mode.AI_PLAY:
                self._ai_play_step()
            else:
                self._update_play()
            self._build_new_frame()
            clock.tick(self._game_speed)
        return 0

    def _update_grid(self):
        for i in range(len(self._env.map)):
            match self._env.map[i]:
                case 'G':
                    self._grid.objs[i].color = self._grid.GREEN_APPEL_COLOR
                case 'H':
                    self._grid.objs[i].color = self._grid.SNAKE_HEAD_COLOR
                case 'O':
                    if i % 2:
                        self._grid.objs[i].color = self._grid.EVEN_COLOR
                    else:
                        self._grid.objs[i].color = self._grid.ODD_COLOR
                case 'R':
                    self._grid.objs[i].color = self._grid.RED_APPLE_COLOR
                case 'S':
                    self._grid.objs[i].color = self._grid.SNAKE_BODY_COLOR
                case 'W':
                    self._grid.objs[i].color = self._grid.WALL_COLOR
                case _:
                    raise UnknowCharInMap()

    def _event_handler(self) -> int:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
            if event.type == pygame.KEYDOWN:
                self._key_handler(event.key)

    def _key_handler(self, pressed_key: int):
        if self._key_handler_general(pressed_key): return
        if self._actual_mode == Mode.HUMAIN_PLAY:
           if self._key_handler_humain_play(pressed_key): return
        if self._actual_mode == Mode.AI_PLAY or self._actual_mode == Mode.AI_TRAIN:
            if self._key_handler_ai_play(pressed_key): return
                
    def _key_handler_general(self, pressed_key: int) -> bool:
        match pressed_key:
            case pygame.K_ESCAPE: pygame.event.post(pygame.event.Event(pygame.QUIT))
            case pygame.K_SPACE: self.toggle_app_status()
            case pygame.K_p: self.increase_game_speed()
            case pygame.K_m: self.decrease_game_speed()
            case pygame.K_r: self.reset_game_speed()
            case _: return False
        return True

    def _key_handler_humain_play(self, pressed_key: int) -> bool:
        match pressed_key:
           case pygame.K_UP: self._env.change_move(Movement.UP)
           case pygame.K_DOWN: self._env.change_move(Movement.DOWN)
           case pygame.K_LEFT: self._env.change_move(Movement.LEFT)
           case pygame.K_RIGHT: self._env.change_move(Movement.RIGHT)
           case _: return False
        return True

    def _key_handler_ai_play(self, pressed_key: int) -> bool:
        match pressed_key:
            case 1073742053: print('PASSAGE EN STEP BY STEP / PASSAGE EN CONTINUE')
            case _: return False
        return True

    def toggle_app_status(self):
        self._status = AppStatus.PLAY if self._status == AppStatus.PAUSE else AppStatus.PAUSE

    def close_app(self):
        self._status = AppStatus.STOP
        pygame.quit()

    def increase_game_speed(self):
        if self._game_speed + GAME_SPEED_FACTOR > GAME_SPEED_MAX:
            logger.warning("Cannot increase the speed game: Max reached.")
            return
        self._game_speed += GAME_SPEED_FACTOR

    def decrease_game_speed(self):
        if self._game_speed - GAME_SPEED_FACTOR < GAME_SPEED_MIN:
            logger.warning("Cannot decrease the speed game: Min reached.")
            return
        self._game_speed -= GAME_SPEED_FACTOR

    def reset_game_speed(self):
        self._game_speed = GAME_SPEED_DEFAULT

    def _ai_train_step(self):
        try:
            next(self._training)
        except StopIteration:
            self._training = None
            logger.info("Training finished.")
            self._status = AppStatus.STOP
            return

    def _ai_play_step(self):
        action: Movement = self._agent.take_action()
        self._env.change_move(action)
        self._update_play()

    def _build_new_frame(self):
        self._update_grid()
        # self._print_map(self._env.snake_view(), self._env.width)
        # logger.info(self._env.actual_move.name)
        self._grid.draw(self._surface)
        pygame.display.flip()

    def _update_play(self):
        try:
            self._env.update()
        except (SnakeLoose, SnakeWin) as exc:
            logger.info(f"Game Finished: {exc.message}, snake size: {self._env.snake.size}")
            self._status = AppStatus.STOP
        except (SnakeGreenApple, SnakeRedApple):
            pass

    def _print_map(self, map: List[str], width:int):
        logger.info('\n')
        for i in range(len(map)):
            print(map[i],end="")
            if not (i+1) % width and i:
                print()
        print()