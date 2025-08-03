import json
from Visualization.functions import plot_loss_and_accuracy, plot_alignment

def visualize(args):
    """
    Visualize the results of the training and evaluation.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
    """
    
    training_metrics_path = f"{args.root_path}/{args.dataset}/{args.model}/{args.attack}/raw_results/train_metrics_{args.attack}_{args.seed}.json"
    evaluation_metrics_path = f"{args.root_path}/{args.dataset}/{args.model}/{args.attack}/raw_results/evaluation_metrics_{args.attack}_{args.seed}.json"

    with open(training_metrics_path, "r") as f:
        training_metrics = json.load(f)

    with open(evaluation_metrics_path, "r") as f:
        evaluation_metrics = json.load(f)

    plot_loss_and_accuracy(args, training_metrics["epoch_metrics"]["loss"], training_metrics["epoch_metrics"]["accuracy"],
                            evaluation_metrics["attack_epoch_metrics"]["loss"], evaluation_metrics["attack_epoch_metrics"]["accuracy"],
                            evaluation_metrics["fgsm_epoch_metrics"]["loss"], evaluation_metrics["fgsm_epoch_metrics"]["accuracy"],
                            evaluation_metrics["pgd_epoch_metrics"]["loss"], evaluation_metrics["pgd_epoch_metrics"]["accuracy"])

    if args.track_alignment:
        plot_alignment(args, training_metrics["alignment_values"])