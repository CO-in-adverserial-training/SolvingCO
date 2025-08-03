import os
import subprocess

datasets = ["CIFAR10", "CIFAR100", "CINIC10", "SVHN", "TinyImageNet", "PathMNIST"]
models = ["PreActResNet18", "ResNet18", "WideResNet28", "SENet18", "VitSmall", "VitBase"]
attacks = ["FGSM", "FGSM-RS", "NFGSM", "ZeroGrad", "TRADES", "PGD"]
root_path = "results"

os.system(f"mkdir {root_path}")
# os.system(f'cd "catastrophic overfitting in adversarial robustness\SolvingCO\Codes"')
EPOCHS = 1

for dataset in datasets:
    for model in models:
        for attack in attacks:
            print(f"Evaluating '{dataset}' dataset, '{model}' model, '{attack}' attack.")

            # output = os.popen(f"python main.py --root_path {root_path} --dataset {dataset} --model {model} --attack {attack} --epochs {EPOCHS}").read()
            output = subprocess.check_output(f"python /kaggle/working/SolvingCO/Codes/main.py --root_path {root_path} --dataset {dataset} --model {model} --attack {attack} --epochs {EPOCHS}", shell=True, text=True)
            print(output)
            
            print("=" * 50)
