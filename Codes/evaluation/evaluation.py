import torch.nn.functional as F
from datasets.get_loaders import get_loaders
from ..utils import load_checkpoint
from attacks.get_attack import get_attack
from ..attacks.attack_params import attack_params_dict
from ..training.utils import MetricTracker

def evaluate(args, device):
    # Get dataset loaders
    _, testloader, upper_limit, lower_limit, _, _, _, num_classes, num_train_samples, num_test_samples = get_loaders(args.dataset)
    # Get attack parameters
    attack_params = attack_params_dict.get(args.attack, {}).copy()

    use_regularizer = args.attack in ["TRADES", "GradAlign", "ELLE"]
    index_dataset = args.attack in ["ATAS", "FGSM-EP"]

    # Setup metric trackers
    test_loss_tracker = MetricTracker() # Track test loss
    test_regularizer_tracker = MetricTracker()

    for epoch in range(args.epochs + 1):
        model, _, _ = load_checkpoint(args.model, num_classes, f"{args.root_path}/checkpoints/model{str(epoch).zfill(3)}.pt")
        model.eval()

        for i, data in enumerate(testloader):
            if index_dataset:
                images, labels, index = data[0].to(device), data[1].to(device), data[2]
            else:
                images, labels = data[0].to(device), data[1].to(device)
            # Determine attack
            attack = get_attack(args.attack)
            match args.attack:
                case attack if attack in  ["FGSM", "FGSM-RS", "NFGSM", "ZeroGrad", "SIA", "PGD"]:
                    delta, _ = attack(model, images, labels, upper_limit, lower_limit, **attack_params)
                case attack if attack in ["TRADES", "GradAlign", "ELLE"]:
                    delta, reg, _ = attack(model, images, labels, upper_limit, lower_limit, **attack_params)
                case "ATAS":
                    delta, _ = attack(model, images, labels, upper_limit, lower_limit, **attack_params)
                case "FGSM-EP":
                    delta, reg, _ = attack(model, images, labels, upper_limit, lower_limit, **attack_params)
                case _:
                    raise "Invalid Attack Method!"
            # Add perturbation to original images
            adv_images = images + delta
            # Forward pass with adversarial examples
            preds = model(adv_images)
            loss = F.cross_entropy(preds, labels)

            #Track Regularizer Value Per Batch
            if use_regularizer:
                reg = reg.cpu() if reg is not None else None
                reg = reg.item()
                test_regularizer_tracker.update(batch_test_reg=reg)
            test_loss_tracker.update(batch_test_loss=loss)