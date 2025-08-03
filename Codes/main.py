from training.train import train
from evaluation.evaluation import evaluate
from utils import create_directories, get_device, set_seeds
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "CINIC10", "SVHN", "TinyImageNet", "PathMNIST"], default="CIFAR10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--normalize_dataset", action="store_true")
    parser.add_argument("--model", choices=["PreActResNet18", "ResNet18", "WideResNet28", "SENet18", "VitSmall", "VitBase"], default="PreActResNet18")
    parser.add_argument("--attack", choices=["SIA", "FGSM", "FGSM-RS", "NFGSM", "ZeroGrad", "ATAS", "TRADES", "PGD"], required=True)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--initial_lr", type=float, default=0.01, help="May be overwritten by scheduler")
    parser.add_argument("--optimizer", choices=["SGD"], default="SGD")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["Cyclic", "MultiStep", "CosineAnnealing"], default="Cyclic")
    parser.add_argument("--track_alignment", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    # Create nescessary directories
    create_directories(args.root_path)
    # Get device
    device = get_device(args.device)
    # Set seed
    set_seeds()
    # Train model
    train(args, device)
    # Evaluate training
    evaluate(args, device)

if __name__ == "__main__":
    main()