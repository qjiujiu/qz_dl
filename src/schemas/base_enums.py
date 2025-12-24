from enum import Enum

class TaskType(str, Enum):
    NLP = "nlp"
    VISION = "vision"


class OptimizerType(str, Enum):
    ADAM  = "adam"
    ADAMW = "adamw"
    SGD   = "sgd"

class SchedulerType(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear" 


class LossType(str, Enum):
    CROSS_ENTROPY = "ce"
    FOCAL_LOSS = "focal"
