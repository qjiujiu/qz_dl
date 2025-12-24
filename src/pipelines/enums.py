from enum import Enum

class TaskType(str, Enum):
    NLP = "nlp"
    VISION = "vision"
    

class OptimizerType(str, Enum):
    ADAM  = "adam"
    ADAMW = "adamw"
    SGD   = "sgd"


class AttenType(str, Enum):
    MLP = "mlp"
    SELF = "self"
    PE = "pe"
    SELF_PE = "self-pe"
    


class AttackType(str, Enum):
    FGSM = "fgsm"
    PGD  = "pgd"