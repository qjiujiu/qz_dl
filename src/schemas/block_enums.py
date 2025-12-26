from enum import Enum

class PluginType(str, Enum):
    MlpAtten = "mlp"
    PosEnc = "pe"
    SA = "self"
    SelfPE = "self-pe"
    ID = "identity"
    GaussLinf = "gauss"
    
    

class AttackType(str, Enum):
    FGSM = "fgsm"
    PGD  = "pgd"