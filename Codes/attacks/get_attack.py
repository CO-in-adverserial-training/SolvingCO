from .fgsm import fgsm
from .trades import trades
from .fgsm_rs import fgsm_rs
from .grad_align import grad_align
from .atas import atas
from .zerograd import zero_grad
from .multigrad import multi_grad
from .nfgsm import nfgsm
from .aaer import fgsm as fgsm_aae
from .elle import elle
from .sia import sia
from .pgd import pgd
from .pgd2 import pgd2

def get_attack(attack_name: str):
    """
    Get the attack for the given attack name.
    
    Args:
        attack_name (str): Name of the attack to get.
    """
    
    match attack_name:
        case "Benign":
            return None
        case "FGSM":
            return fgsm
        case "TRADES":
            return trades
        case "FGSM-RS":
            return fgsm_rs
        case "GradAlign":
            return grad_align
        case "ATAS":
            return atas
        case "ZeroGrad":
            return zero_grad
        case "MultiGrad":
            return multi_grad
        case "NFGSM":
            return nfgsm
        case "AAER":
            return fgsm_aae
        case "ELLE":
            return elle
        case "SIA":
            return sia
        case "PGD":
            return pgd
        case "PGD2":
            return pgd2
        case _:
            raise ValueError('Invalid Attack!')
        
