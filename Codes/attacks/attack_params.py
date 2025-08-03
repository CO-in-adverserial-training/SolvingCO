# Stores attack specific hyperparameters
def get_attack_params(epsilon: float):
    return {
        "SIA": {
            "epsilon": epsilon,
            "alpha": 2 * epsilon
        },
        "FGSM": {
            "epsilon": epsilon,
            "alpha": 2 * epsilon
        },
        "FGSM-RS": {
            "epsilon": epsilon,
            "alpha": 1.25 * epsilon,
            "k": 1.0
        },
        "NFGSM": {
            "epsilon": epsilon,
            "alpha": epsilon,
            "k": 2.0
        },
        "PGD": {
            "epsilon": epsilon,
            "alpha": epsilon / 4,
            "attack_iters": 10,
            "k": 1.0,
            "clip": True
        },
        "TRADES": {
            "epsilon": epsilon,
            "perturb_steps": 10,
            "step_size": 0.007
        },
        "GradAlign": {
            "epsilon": epsilon,
            "alpha": epsilon,
            "k": 1.0
        },
        "ELLE": {
            "epsilon": epsilon,
            "alpha": epsilon,
            "k": 1.0
        },
        "ATAS": {
            "epsilon": epsilon,
            "alpha": epsilon / 4,
            "topk": 3,
            "num_steps": 5
        },
        "FGSM-EP": {
            "epsilon": epsilon,
            "alpha": 1.0
        },
    }

def get_regularizer_params(epsilon: float):
    return {
        "TRADES": {
            "reg": 6.0 # Beta
        },
        "GradAlign": {
            "reg": 0.2 # Lambda Alignment
        },
        "ELLE": {
            "reg": 1.0 # Lambda ELLE
        },
    }
