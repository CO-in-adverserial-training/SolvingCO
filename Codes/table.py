import json
import numpy as np
from matplotlib import pyplot as plt

SEEDS = [21, 33, 42]
DATASET = 'CIFAR10'
MODEL = 'PreActResNet18'
METHODS = ['FGSM', 'FGSM-RS', 'GradAlign', 'ZeroGrad', 'MultiGrad', 'NFGSM', 'AAER', 'ELLE', 'TRADES']
RESULTS_DIR = '/home/frahmani/Github Code/SolvingCO/Codes/results/Results'


def print_latex_table():
    for method in METHODS:
        benign, fgsm, pgd = [], [], []
        for seed in SEEDS:
            eval_metrics = f'{RESULTS_DIR}/{DATASET}/{MODEL}/{method}/raw_results_{seed}/evaluation_metrics.json'
            with open(eval_metrics, 'r') as f:
                data = json.load(f)

            benign.append(data['benign_epoch_metrics']['accuracy'][-1])
            fgsm.append(data['fgsm_epoch_metrics']['accuracy'][-1])
            pgd.append(data['pgd_epoch_metrics']['accuracy'][-1])
        benign, fgsm, pgd = 100 * np.array(benign), 100 * np.array(fgsm), 100 * np.array(pgd)
        b_mu, b_std = np.mean(benign), np.std(benign)
        f_mu, f_std = np.mean(fgsm), np.std(fgsm)
        p_mu, p_std = np.mean(pgd), np.std(pgd)
        print(f'{method} & {b_mu:.2f} ± {b_std:.2f} \\% & {f_mu:.2f} ± {f_std:.2f} \\% & {p_mu:.2f} ± {p_std:.2f} \\% & AA \\\\')



if __name__ == '__main__':
    print_latex_table()