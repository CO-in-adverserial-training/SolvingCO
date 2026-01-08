# Stores attack specific hyperparameters

attack_params_dict = {
    "SORA": {
        "epsilon": 8 / 255,
        "alpha": 16/ 255
    },
    "FGSM": {
        "epsilon": 8 / 255,
        "alpha": 16/ 255
    },
    "FGM": {
        "epsilon": 1.0,
        "alpha": 2.0
    },
    "FGSM-RS": {
        "epsilon": 8 / 255,
        "alpha": 10/ 255,
        "k": 1.0
    },
    "FGM-RS": {
        "epsilon": 1.0,
        "alpha": 1.25,
        "k": 1.0
    },
    "NFGSM": {
        "epsilon": 8 / 255,
        "alpha": 8 / 255,
        "k": 2.0
    },
    "PGD": {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "attack_iters": 10,
        "k": 1.0,
        "clip": True
    },
    "PGD2": {
        "epsilon": 8 / 255,
        "alpha": 4 / 255,
        "attack_iters": 2,
        "k": 1.0,
        "clip": True
    },
    "TRADES": {
        "epsilon": 8 / 255,
        "perturb_steps": 10,
        "step_size": 0.007
    },
    "GradAlign": {
        "epsilon": 8 / 255,
        "alpha": 8 / 255,
        "k": 1.0
    },
    "ELLE": {
        "epsilon": 8 / 255,
        "alpha": 8 / 255,
        "k": 1.0
    },
    "ATAS": {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "topk": 3,
        "num_steps": 5
    },
    "FGSM-EP": {
        "epsilon": 8 / 255,
        "alpha": 1.0
    },
}

regularizer_params_dict = {
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
