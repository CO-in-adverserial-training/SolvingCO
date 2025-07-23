import torch.nn.functional as F
from attacks.get_attack import get_attack
from architectures.get_model import get_model
from .alignment import calc_alignment
from ..utils import save_checkpoint

def train(dataloader, model_name, num_classes: int, attack_name: str, attack_params, optimizer, scheduler, num_epochs: int=30, device:str="cuda"):
    model = get_model(model_name, num_classes)
    use_regularizer = attack_name in ["TRADES", "GradAlign", "ELLE"]
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            if index_dataset:
                images, labels, index = data[0].to(device), data[1].to(device), data[2]
            else:
                images, labels = data[0].to(device), data[1].to(device)
            # Zero out previous gradient accumulation
            optimizer.zero_grad()
            # Determine attack
            attack = get_attack(attack_name)
            match attack_name:
                case attack if attack in  ["FGSM", "FGSM-RS", "NFGSM", "ZeroGrad", "SIA", "PGD"]:
                    delta, grad = attack(model, images, labels, **attack_params)
                case attack if attack in ["TRADES", "GradAlign", "ELLE"]:
                    delta, reg, grad = attack(model, images, labels, **attack_params)
                case "ATAS":
                    delta, grad = attack(model, images, labels, index, **attack_params)
                    delta_atas[index] = delta.clone().detach()
                case "FGSM-EP":
                    delta, reg, grad = attack(model, images, labels, index, **attack_params)
                    delta_fgsm_ep[index] = delta.clone().detach()
                case _:
                    raise "Invalid Attack Method!"
            
            # Add perturbation to original images
            adv_images = images + delta
            if calculate_alignment:
                adv_images.requires_grad = True
            # Forward pass with adversarial examples
            preds = model(adv_images)
            loss = F.cross_entropy(preds, labels)
            # Add regularization term if needed
            if use_regularizer:
                loss += reg_coeff * reg
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()
            # Update Scheduler # TODO this is only for Cyclic LR not for MultiStep
            scheduler.step()
            if calculate_alignment:
                alignment = calc_alignment(grad, adv_images)


    # Save training checkpoint
    save_checkpoint(model, optimizer, scheduler, f"{root_path}/checkpoints/model{str(epoch).zfill(3)}.pt")


