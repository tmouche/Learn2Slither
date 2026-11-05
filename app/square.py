from typing import List
from pygame import Surface, Rect
import pygame

class Square():

    color: str
    
    _pos: List[int]
    _obj: Rect
    _size: int

    def __init__(self, size: int, pos: List[int], color: str):
        self.color = color
        self._size = size
        self._pos = pos
        self._obj = Rect(
            (pos[0], pos[1]),
            (size, size)
        )

    @property
    def size(self) -> int:
        return self._size

    def draw(self, surface: Surface):
        pygame.draw.rect(
            surface=surface,
            color=self.color,
            rect=self._obj
        )
