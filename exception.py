
from logger import logger

MAP_TO_LARGE_ERROR = "map too big"
MAP_TO_SMALL_ERROR = "map too small"
WINDOW_TO_LARGE_ERROR = "window too big"
UNKNOWN_CHAR_IN_MAP = "Unknown char in map"
FILE_INIT_FAIL = "Agent initialisation with file failed: "
UNEXPECTED = "Unexpected Exception catched"
FORMAT = "Parsing issue with format: "

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

class AgentFileInitFail(CriticalException):
    def __init__(self, path_to_file: str):
        super().__init__(FILE_INIT_FAIL + path_to_file)

class UnexpectedException(CriticalException):
    def __init__(self):
        super().__init__(UNEXPECTED)

class Format(CriticalException):
    def __init__(self, context: str):
        super().__init__(FORMAT + context)

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