
from logger import logger

MAP_TO_LARGE_ERROR = "The map gave is too big, please reduce"
MAP_TO_SMALL_ERROR = "The map gave in too small, please increase"

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

