import tkinter

class Menu(tkinter.Canvas):

    width: int
    height: int

    human_frame: tkinter.Frame
    human_button: tkinter.Button
    ia_train_frame: tkinter.Frame
    ia_train_button: tkinter.Button
    ia_play_frame: tkinter.Frame
    ia_play_button: tkinter.Button
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
        self.ia_train_frame = None
        self.ia_train_button = None
        self.ia_play_frame = None
        self.ia_play_button = None
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

    def create_ia_train_start_button(self, cmd_fnc:callable):
        self.ia_train_frame = tkinter.Frame(
            self,
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT,
        )
        self.ia_train_frame.pack_propagate(False)
        self.ia_train_frame.pack()
        self.ia_train_button = tkinter.Button(
            self.ia_train_frame,
            text="Train IA",
            command=cmd_fnc,
            bg=self.COLOR_ACTIVATE_BUTTON
        )
        self.ia_train_button.pack(fill="both", expand=True, side="left")

    def create_ia_play_start_button(self, cmd_fnc:callable):
        self.ia_play_frame = tkinter.Frame(
            self,
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT,
        )
        self.ia_play_frame.pack_propagate(False)
        self.ia_play_frame.pack()
        self.ia_play_button = tkinter.Button(
            self.ia_play_frame,
            text="Start IA",
            command=cmd_fnc,
            bg=self.COLOR_ACTIVATE_BUTTON
        )
        self.ia_play_button.pack(fill="both", expand=True, side="left")

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
        """Literaly do nothing"""
        pass