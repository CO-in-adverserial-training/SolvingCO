import json
from Visualization.functions import plot_loss_and_accuracy, plot_alignment, plot_accs_vs_eps

def visualize(args):
    """
    Visualize the results of the training and evaluation.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
    """
    

    training_metrics_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results/train_metrics_{args.seed}.json"
    with open(training_metrics_path, "r") as f:
        training_metrics = json.load(f)

    evaluation_metrics_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results/evaluation_metrics_{args.seed}.json"
    with open(evaluation_metrics_path, "r") as f:
        evaluation_metrics = json.load(f)

    accs_vs_eps_metrics_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results/accs_vs_epps_metrics_{args.seed}.json"
    with open(accs_vs_eps_metrics_path, "r") as f:
        accs_vs_eps_metrics = json.load(f)


    plot_loss_and_accuracy(args, training_metrics["epoch_metrics"]["loss"], training_metrics["epoch_metrics"]["accuracy"],
                            evaluation_metrics["attack_epoch_metrics"]["loss"], evaluation_metrics["attack_epoch_metrics"]["accuracy"],
                            evaluation_metrics["benign_epoch_metrics"]["loss"], evaluation_metrics["benign_epoch_metrics"]["accuracy"],
                            evaluation_metrics["fgsm_epoch_metrics"]["loss"], evaluation_metrics["fgsm_epoch_metrics"]["accuracy"],
                            evaluation_metrics["pgd_epoch_metrics"]["loss"], evaluation_metrics["pgd_epoch_metrics"]["accuracy"])

    if args.track_alignment:
        plot_alignment(args, training_metrics["alignment_values"])


    plot_accs_vs_eps(args, accs_vs_eps_metrics["accs_vs_epps_metrics"]["fgsm_accs"], accs_vs_eps_metrics["accs_vs_eps_metrics"]["pgd_accs"],
                      accs_vs_eps_metrics["accs_vs_epps_metrics"]["clean_accs"])