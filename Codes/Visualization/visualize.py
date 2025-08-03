from Visualization.functions import plot_loss_and_accuracy, plot_alignment

def visualize(args):
    training_metrics_path = f"{args.root_path}/raw_results/train_metrics_{args.attack}_{args.seed}.json"
    evaluation_metrics_path = f"{args.root_path}/raw_results/evaluation_metrics_{args.attack}_{args.seed}.json"

    plot_loss_and_accuracy(training_metrics_path)

    plot_loss_and_accuracy(evaluation_metrics_path)

    if args.track_alignment:
        plot_alignment(training_metrics_path)