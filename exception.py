
from logger import logger

MAP_TO_LARGE_ERROR = "map too big"
MAP_TO_SMALL_ERROR = "map too small"
WINDOW_TO_LARGE_ERROR = "window too big"
UNKNOWN_CHAR_IN_MAP = "Unknown char in map"

class CriticalException(Exception):

    message: str

    def __init__(self, context:str):
        self.message = context

class FunctionalException(Exception):

    message: str

    def __init__(self, context: str):
        self.message = context

#-------------------------------------------#
#           --CRITICAL EXCEPTION--          #
#-------------------------------------------#

class MapToLarge(CriticalException):
    def __init__(self):
        super().__init__(MAP_TO_LARGE_ERROR)
    
class MapToSmall(CriticalException):
    def __init__(self):
        super().__init__(MAP_TO_SMALL_ERROR)

class WindowToLarge(CriticalException):
    def __init__(self):
        super().__init__(WINDOW_TO_LARGE_ERROR)

class UnknowCharInMap(CriticalException):
    def __init__(self):
        super().__init__(UNKNOWN_CHAR_IN_MAP)


#-------------------------------------------#
#         --FUNCTIONNAL EXCEPTION--         #
#-------------------------------------------#

class SnakeLoose(FunctionalException):
    def __init__(self, detail:str):
        super().__init__(f"Snake lose from: {detail}")

class SnakeWin(FunctionalException):
    def __init__(self):
        super().__init__(f"Snake win")

class SnakeGreenApple(FunctionalException):
    def __init__(self):
        super().__init__(f"Snake has eaten a green apple")

class SnakeRedApple(FunctionalException):
    def __init__(self):
        super().__init__(f"Snake has eaten a red apple")