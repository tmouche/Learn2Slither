
from logger import logger

MAP_TO_LARGE_ERROR = "map too big"
MAP_TO_SMALL_ERROR = "map too small"
WINDOW_TO_LARGE_ERROR = "window too big"
UNKNOWN_CHAR_IN_MAP = "Unknown char in map"

class CriticalException(Exception):
    def __init__(self, context:str):
        logger.error(context)

class FunctionalException(Exception):
    def __init__(self):
        pass

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