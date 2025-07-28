import torch
import torch.nn.functional as F
import time
import json
from datasets.get_loaders import get_loaders
from architectures.get_model import get_model
from attacks.get_attack import get_attack
from attacks.attack_params import attack_params_dict, regularizer_params_dict
from utils import save_checkpoint
from training.alignment import calc_alignment
from training.utils import MetricTracker, get_optimizer, get_scheduler, get_input_dimensions, calculate_batch_accuracy

def train(args, device):
    index_dataset = args.attack in ["ATAS", "FGSM-EP"]
    # Get dataset loaders
    trainloader, _, upper_limit, lower_limit, _, _, _, num_classes, num_train_samples, num_test_samples = get_loaders(args, index_dataset, device)
    # Get model
    model = get_model(args.model, num_classes)
    model = model.to(device)
    model.train()
    # Get optimizer
    optimizer = get_optimizer(args, model)
    # Get scheduler
    scheduler = get_scheduler(args, optimizer, len(trainloader))
    # Determine attack
    attack = get_attack(args.attack)
    # Get attack parameters
    attack_params = attack_params_dict.get(args.attack, {}).copy()
    # Get regularization coefficient if needed
    use_regularizer = args.attack in ["TRADES", "GradAlign", "ELLE", "FGSM-EP"]
    if use_regularizer:
        reg_params = regularizer_params_dict.get(args.attack, {}).copy()

    if index_dataset:
        _, C, H, W = get_input_dimensions(trainloader, index_dataset)
        delta = torch.zeros((num_train_samples, C, H, W), device=device)
        delta.uniform_(-args.epsilon, args.epsilon)
        attack_params["delta"] = delta

    # Save initial checkpoint
    save_checkpoint(model, optimizer, scheduler, f"{args.root_path}/checkpoints/model{str(0).zfill(3)}.pt")
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
                    raise ValueError("Invalid Attack Method!")
            
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
            # Update scheduler
            if args.scheduler in ["Cyclic", "CosineAnnealing"]:
                scheduler.step()
            
            if args.track_alignment:
                alignment = calc_alignment(grad, adv_images)
                if args.attack == "SIA":
                    attack_params["alignment"] = alignment # Save as attack param to use in the next batch for SIA
                alignment_tracker.update(batch_alignment=alignment.item())
            #Track Regularizer Value Per Batch
            if use_regularizer:
                reg = reg.cpu().item() if reg is not None else 0.0
                regularizer_tracker.update(batch_train_reg=reg)

            batch_accuracy = calculate_batch_accuracy(preds, labels)
            batch_tracker.update(loss=loss.item(), accuracy=batch_accuracy.item())
            alpha_tracker.update(batch_alpha=attack_params["alpha"])

        if args.scheduler in ["MultiStep"]:
            scheduler.step()
        
        epoch_loss = batch_tracker.average("loss")
        epoch_accuracy = batch_tracker.average("accuracy")
        
        finish_time = time.time()
        epoch_time = finish_time - start_time
        total_train_time += epoch_time
        # Print epoch loss and accuracy
        print(f"Epoch {epoch} - Loss {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2%}, Time {epoch_time:.4f}")
        epoch_tracker.update(loss=epoch_loss, accuracy=epoch_accuracy)
        batch_tracker.reset()

        # Save training checkpoint
        save_checkpoint(model, optimizer, scheduler, f"{args.root_path}/checkpoints/model{str(epoch).zfill(3)}.pt")

    # Save training metrics for processing and visualization
    metrics_to_save = {
        "epoch_metrics": epoch_tracker.to_dict(),
        "alignment_values": alignment_tracker.to_dict(),
        "alpha_values": alpha_tracker.to_dict(),
        "regularizer_values": regularizer_tracker.to_dict()
    }

    with open(f"train_metrics_{args.attack}.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print('Finished Training')
    print("Total Training Time: ", total_train_time)
