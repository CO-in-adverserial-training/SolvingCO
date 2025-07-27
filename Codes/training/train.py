import torch
import torch.nn.functional as F
from datasets.get_loaders import get_loaders
from architectures.get_model import get_model
from attacks.get_attack import get_attack
from ..attacks.attack_params import attack_params_dict, regularizer_params_dict
from .alignment import calc_alignment
from ..utils import save_checkpoint, get_device
from .utils import get_optimizer, get_scheduler, get_input_dimensions

def train(args, device):
    # Get dataset loaders
    trainloader, _, upper_limit, lower_limit, _, _, _, num_classes, num_train_samples, num_test_samples = get_loaders(args.dataset)
    # Get model
    model = get_model(args.model, num_classes)
    # Get optimizer
    optimizer = get_optimizer(args.optimizer, model)
    # Get scheduler
    scheduler = get_scheduler(args.scheduler, optimizer, len(trainloader))
    # Determine attack
    attack = get_attack(args.attack)
    # Get attack parameters
    attack_params = attack_params_dict.get(args.attack, {}).copy()
    # Get regularization coefficient if needed
    use_regularizer = args.attack in ["TRADES", "GradAlign", "ELLE"]
    if use_regularizer:
        reg_params = regularizer_params_dict.get(args.attack, {}).copy()

    index_dataset = args.attack in ["ATAS", "FGSM-EP"]
    if index_dataset:
        _, C, H, W = get_input_dimensions(trainloader, index_dataset)
        delta = torch.zeros((num_train_samples, C, H, W), device=device)
        delta.uniform_(-args.epsilon, args.epsilon)
        attack_params["delta"] = delta

    for epoch in range(args.epochs):
        for i, data in enumerate(trainloader):
            if index_dataset:
                images, labels, index = data[0].to(device), data[1].to(device), data[2]
            else:
                images, labels = data[0].to(device), data[1].to(device)
            # Zero out previous gradient accumulation
            optimizer.zero_grad()
            match args.attack:
                case attack if attack in  ["FGSM", "FGSM-RS", "NFGSM", "ZeroGrad", "SIA", "PGD"]:
                    delta, grad = attack(model, images, labels, upper_limit, lower_limit, **attack_params)
                case attack if attack in ["TRADES", "GradAlign", "ELLE"]:
                    delta, reg, grad = attack(model, images, labels, upper_limit, lower_limit, **attack_params)
                case "ATAS":
                    delta, grad = attack(model, images, labels, index, upper_limit, lower_limit, **attack_params)
                    delta[index] = delta.clone().detach()
                case "FGSM-EP":
                    delta, reg, grad = attack(model, images, labels, index, upper_limit, lower_limit, **attack_params)
                    delta[index] = delta.clone().detach()
                case _:
                    raise "Invalid Attack Method!"
            
            # Add perturbation to original images
            adv_images = images + delta
            if args.track_alignment:
                adv_images.requires_grad = True
            # Forward pass with adversarial examples
            preds = model(adv_images)
            loss = F.cross_entropy(preds, labels)
            # Add regularization term if needed
            if use_regularizer:
                loss += reg_params["reg"] * reg
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()
            # Update Scheduler # TODO this is only for Cyclic LR not for MultiStep
            scheduler.step()
            if args.track_alignment:
                alignment = calc_alignment(grad, adv_images)
                if args.attack == "SIA":
                    attack_params["alignment"] = alignment # Save as attack param to use in the next batch for SIA
    # Save training checkpoint
    save_checkpoint(model, optimizer, scheduler, f"{args.root_path}/checkpoints/model{str(epoch).zfill(3)}.pt")


