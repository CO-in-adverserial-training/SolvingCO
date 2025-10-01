# Stores attack specific hyperparameters
def get_attack_params(args):
    match args.dataset:
        case "CIFAR10":
            q_val = 0.35
        case "CIFAR100":
            q_val = 0.45
        case "SVHN":
            q_val = 0.7
        case _:
            q_val = 0.45
    
    return {
        "SIA": {
            "epsilon": args.epsilon,
            "max_alpha": 2 * args.epsilon,
            "method": "Second Order Theory Sign"
        },
        "Benign": {
            "epsilon": 0,
            "alpha": 0
        },
        "FGSM": {
            "epsilon": args.epsilon,
            "alpha": 2 * args.epsilon
        },
        "FGSM-RS": {
            "epsilon": args.epsilon,
            "alpha": 1.25 * args.epsilon,
            "k": 1.0
        },
        "NFGSM": {
            "epsilon": args.epsilon,
            "alpha": args.epsilon,
            "k": 2.0
        },
        "ZeroGrad": {
            "epsilon": args.epsilon,
            "alpha": 2 * args.epsilon,
            "q_val": q_val,
            "k": 1.0,
            "clip": True
        },
        "MultiGrad": {
            "epsilon": args.epsilon,
            "alpha": 2 * args.epsilon,
            "samples": 3,
            "zeroing_th": -1,
            "k": 1.0,
            "parallel": True
        },
        "PGD": {
            "epsilon": args.epsilon,
            "alpha": args.epsilon / 4,
            "attack_iters": 10,
            "k": 1.0,
            "clip": True
        },
        "PGD2": {
            "epsilon": args.epsilon,
            "alpha": args.epsilon / 2,
            "attack_iters": 2,
            "k": 1.0,
            "clip": True
        },
        "TRADES": {
            "epsilon": args.epsilon,
            "perturb_steps": 10,
            "alpha": args.epsilon / 4
        },
        "GradAlign": {
            "epsilon": args.epsilon,
            "alpha": 1.25 * args.epsilon,
            "k": 1.0
        },
        "ELLE": {
            "epsilon": args.epsilon,
            "alpha": args.epsilon,
            "k": 1.0
        },
        "AAER": {
            "epsilon": args.epsilon,
            "alpha": args.epsilon,
            "k": 2.0,
            "clip": False
        },
        "ATAS": {
            "epsilon": args.epsilon,
            "beta": 0.5,
            "gamma_over_c": 2 * args.epsilon,
            "c": 0.01,
            "min_step_size": 0.5 * args.epsilon,
            "max_step_size": 1.75 * args.epsilon,
            "warm_up_epoch": 5
        },
    }

def get_regularizer_params(args):
    match args.dataset:
        case "CIFAR10":
            grad_align_reg = 0.2
        case "SVHN":
            grad_align_reg = 2.5
        case _:
            grad_align_reg = 0.2
    
    return {
        "TRADES": {
            "reg": 6.0 # Beta
        },
        "GradAlign": {
            "reg": grad_align_reg # Lambda Alignment
        },
        "ELLE": {
            "reg": 1000 # Lambda ELLE
        },
        "AAER": {
            "lambda1": 1.0,
            "lambda2": 1.5,
            "lambda3": 0.15,
        },
    }
