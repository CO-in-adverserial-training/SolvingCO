import torch
import torch.nn.functional as F
import time
import json
from datasets.get_loaders import get_loaders
from architectures.get_model import get_model
from attacks.get_attack import get_attack
from attacks.aaer import aaer
from attacks.attack_params import get_attack_params, get_regularizer_params
from utils import save_checkpoint
from training.alignment import calc_alignment
from training.utils import MetricTracker, get_optimizer, get_scheduler, get_input_dimensions, calculate_batch_corrects

def train(args, device):
    """
    Train the model.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        device (torch.device): Device to use for the training.
    """
    
    index_dataset = args.attack in ["ATAS", "FGSM-EP"]
    # Get dataset loaders
    trainloader, _, upper_limit, lower_limit, mu, std, _, num_classes, num_train_samples, num_test_samples = get_loaders(args, index_dataset, device)
    _, C, H, W = get_input_dimensions(trainloader, index_dataset)
    # Get model
    model = get_model(args.model, num_classes, H, C)
    model = model.to(device)
    model.train()
    # Get optimizer
    optimizer = get_optimizer(args, model)
    # Get scheduler
    scheduler = get_scheduler(args, optimizer, len(trainloader))
    # Determine attack
    attack = get_attack(args.attack)
    # Get attack parameters
    attack_params = get_attack_params(args.epsilon).get(args.attack, {}).copy()
    # Get regularization coefficient if needed
    use_regularizer = args.attack in ["TRADES", "GradAlign", "ELLE"]
    if use_regularizer:
        reg_params = get_regularizer_params(args.epsilon).get(args.attack, {}).copy()

    if index_dataset:
        delta = torch.empty((num_train_samples, C, H, W), device=device).uniform_(-1, 1) * (args.epsilon / std).view(1, -1, 1, 1)
        attack_params["delta"] = delta

    if args.attack == "ATAS":
        moving_grad_norm = torch.zeros(num_train_samples, device=device)
        attack_params["moving_grad_norm"] = moving_grad_norm

    # Save initial checkpoint
    save_checkpoint(model, optimizer, scheduler, f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/checkpoints_{args.seed}/model{str(0).zfill(3)}.pt")
    # Setup metric trackers
    batch_tracker = MetricTracker() # Track each batch accuracy and loss
    epoch_tracker = MetricTracker() # Track each epoch accuracy and loss
    alignment_tracker = MetricTracker() # Track alignment
    alpha_tracker = MetricTracker() # Track attack step sizes
    regularizer_tracker = MetricTracker() # Track regularizer vlue

    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        for i, data in enumerate(trainloader):
            if index_dataset:
                images, labels, index = data[0].to(device), data[1].to(device), data[2]
            else:
                images, labels = data[0].to(device), data[1].to(device)
            # Zero out previous gradient accumulation
            optimizer.zero_grad()
            match args.attack:
                case args.attack if args.attack in  ["FGSM", "FGSM-RS", "NFGSM", "ZeroGrad", "MultiGrad", "PGD"]:
                    delta, grad = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case args.attack if args.attack in ["TRADES", "GradAlign", "ELLE"]:
                    delta, reg, grad = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case "SIA":
                    delta, grad, alpha = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case "AAER":
                    delta, grad, clean_logit, loss_before = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case "ATAS":
                    attack_params["warm_up"] = epoch <= attack_params["warm_up_epoch"]
                    delta, grad, moving_grad_norm, alpha = attack(model, images, labels, index, upper_limit, lower_limit, mu, std, **attack_params)
                    attack_params["delta"][index] = delta.detach()
                    attack_params["moving_grad_norm"][index] = moving_grad_norm.detach()
                case "FGSM-EP":
                    delta, reg, grad = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                    attack_params["delta"][index] = delta.detach()
                case _:
                    raise ValueError("Invalid Attack Method!")

            optimizer.zero_grad()

            if args.track_alignment:
                delta.requires_grad = True
            # Add perturbation to original images
            adv_images = images + delta
            # Forward pass with adversarial examples
            preds = model(adv_images)
            loss = F.cross_entropy(preds, labels)
        
            if args.attack == "AAER":
                loss = aaer(loss_before, clean_logit, preds, labels)
        
            # Add regularization term if needed
            if use_regularizer:
                loss += reg_params["reg"] * reg
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()
            # Update scheduler
            if args.scheduler in ["Cyclic", "CosineAnnealing"]:
                scheduler.step()
            
            if args.track_alignment:
                alignment = calc_alignment(grad, delta)
                if args.attack == "SIA":
                    attack_params["alignment"] = alignment # Save as attack param to use in the next batch for SIA
                alignment_tracker.update(batch_alignment=alignment)
            #Track Regularizer Value Per Batch
            if use_regularizer:
                reg = reg.cpu().item() if reg is not None else 0.0
                regularizer_tracker.update(batch_train_reg=reg)

            batch_corrects = calculate_batch_corrects(preds, labels)
            batch_tracker.update(loss=loss.item(), accuracy=batch_corrects.item())
            alpha_tracker.update(batch_alpha=attack_params["alpha"] if args.attack not in ["ATAS", "SIA"] else alpha)

        if args.scheduler in ["MultiStep"]:
            scheduler.step()
        
        epoch_loss = batch_tracker.average("loss")
        epoch_accuracy = batch_tracker.sum("accuracy") / num_train_samples
        
        finish_time = time.time()
        epoch_time = finish_time - start_time
        total_train_time += epoch_time
        # Print epoch loss and accuracy
        print(f"Epoch {epoch} - Loss {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2%}, Time {epoch_time:.4f}")
        epoch_tracker.update(loss=epoch_loss, accuracy=epoch_accuracy)
        batch_tracker.reset()

        # Save training checkpoint
        save_checkpoint(model, optimizer, scheduler, f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/checkpoints_{args.seed}/model{str(epoch).zfill(3)}.pt")

    # Save last checkpoint separately for fututure evaluation
    save_checkpoint(model, optimizer, scheduler, f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/final_checkpoints_{args.seed}/model{str(args.epochs).zfill(3)}.pt")
    # Save training metrics for processing and visualization
    metrics_to_save = {
        "epoch_metrics": epoch_tracker.to_dict(),
        "alignment_values": alignment_tracker.to_dict(),
        "alpha_values": alpha_tracker.to_dict(),
        "regularizer_values": regularizer_tracker.to_dict()
    }

    with open(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}/train_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print('Finished Training')
    print("Total Training Time: ", total_train_time)
