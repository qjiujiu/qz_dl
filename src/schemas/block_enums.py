from enum import Enum

class AttenType(str, Enum):
    MLP = "mlp"
    PE = "pe"
    SELF = "self"
    SELF_PE = "self-pe"
    

class AttackType(str, Enum):
    FGSM = "fgsm"
    PGD  = "pgd"