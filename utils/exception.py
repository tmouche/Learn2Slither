UNEXPECTED = "Unexpected Exception catched"
FORMAT = "Parsing issue with format:"

INIT_ERR = "Initialization failed:"

MODEL = "[MODEL]"
MODEL_OPTI = "Optimizer not found:"
MODEL_LOSS = "Loss not found:"
MODEL_ACTIVATION = "Activation not found:"

NETWORK = "[NETWORK]"
NETWORK_LAYER_COUNT = "Count of layer has to be greater than 2"
NETWORK_BATCH_SIZE = "Batch Size has to be greater than 0"
NETWORK_LEARNING_RATE = "Learning Rate has to be greater than 0"

LAYER = "[LAYER]"
LAYER_SHAPE = "Shape less or egale to zero"
LAYER_ACTIVATION = "Layer activation function:"
LAYER_INITIALIZER = "Layer initializer function:"

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

#           --MODEL EXCEPTION--             #
class ModelException(CriticalException):
    def __init__(self, context: str):
        super().__init__(f"{MODEL} {context}")

class ModelNetCreate(ModelException):
    def _init__(self, context: str):
        super().__init__(f"{INIT_ERR} {context}")

class ModelOptimizer(ModelException):
    def __init__(self, context: str):
        super().__init__(f"{MODEL_OPTI} {context}")

class ModelLoss(ModelException):
    def __init__(self, context: str):
        super().__init__(f"{MODEL_LOSS} {context}")

class ModelActivation(ModelException):
    def __init__(self, context: str):
        super().__init__(f"{MODEL_LOSS} {context}")

#           --NETWORK EXCEPTION--           #
class NetworkException(CriticalException):
    def __init__(self, context: str):
        super().__init__(f"{NETWORK} {context}")

class NetworkInit(NetworkException):
    def __init__(self, context: str):
        super().__init__(f"{INIT_ERR} {context}")

class NetworkBatchSize(NetworkInit):
    def __init__(self):
        super().__init__(NETWORK_BATCH_SIZE)

class NetworkLayerCount(NetworkInit):
    def __init__(self):
        super().__init__(NETWORK_LAYER_COUNT)

class NetworkLearningRate(NetworkInit):
    def __init__(self):
        super().__init__(NETWORK_LEARNING_RATE)



#           --LAYER EXCEPTION--             #
class LayerException(CriticalException):
    def __init__(self, context: str):
        super().__init__(f"{LAYER} {context}")

class LayerInit(LayerException):
    def __init__(self, context: str):
        super().__init__(f"{INIT_ERR} {context}")

class LayerActivation(LayerInit):
    def __init__(self, context: str):
        super().__init__(f"{LAYER_ACTIVATION} {context}")

class LayerInitializer(LayerInit):
    def __init__(self, context: str):
        super().__init__(f"{LAYER_INITIALIZER} {context}")

class LayerShape(LayerInit):
    def __init__(self):
        super().__init__(LAYER_SHAPE)


#           --UTILS EXCEPTION--             #
class UnexpectedException(CriticalException):
    def __init__(self):
        super().__init__(UNEXPECTED)

class Format(CriticalException):
    def __init__(self, context: str):
        super().__init__(f"{FORMAT} {context}")




#-------------------------------------------#
#         --FUNCTIONNAL EXCEPTION--         #
#-------------------------------------------#

