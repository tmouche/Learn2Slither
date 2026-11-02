from typing import List
import tkinter

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
