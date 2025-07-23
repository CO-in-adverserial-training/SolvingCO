from .fgsm import fgsm
from .trades import trades
from .fgsm_rs import fgsm_rs
from .grad_align import grad_align
from .zerograd import zero_grad
from .nfgsm import nfgsm
from .elle import elle
from .sia import sia
from .pgd import pgd

def get_attack(attack_name: str):
    match attack_name:
        case "FGSM":
            return fgsm
        case "TRADES":
            return trades
        case "FGSM-RS":
            return fgsm_rs
        case "GradAlign":
            return grad_align
        case "ZeroGrad":
            return zero_grad
        case "NFGSM":
            return nfgsm
        case "ELLE":
            return elle
        case "SIA":
            return sia
        case "PGD":
            return pgd
        
