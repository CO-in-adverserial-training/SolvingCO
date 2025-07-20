import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR


def fgsm(model, x, y, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 1.0, k: float = 2.0, clip: bool = False, device: str = 'cuda'):
    # Initialize random step
    eta = torch.zeros_like(x).to(device)
    if k != 0:
        for j in range(len(epsilon)):
            eta[:, j, :, :].uniform_(-k * epsilon[j][0][0].item(), k * epsilon[j][0][0].item())
        eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True
    
    output = model(x + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0]
    grad = grad.detach()
    
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * epsilon * torch.sign(grad)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    if clip:
        delta = torch.clamp(delta, -epsilon, +epsilon)
    delta = delta.detach()
    
    return delta, output.detach(), grad


def train_epoch(model, optimizer, scheduler, loader, criterion, attack_params, lambda1: float = 1.0, lambda2: float = 4.0, eval_mode_attack: bool = False, device: str = 'cuda'):
    batch_losses = []
    batch_accuracies = []
    batch_alignments = []
    total, correct = 0, 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        # Attack
        if eval_mode_attack:
            model.eval()
            delta, clean_logits, grad = fgsm(model, img, lbl, **attack_params, device=device)
            model.train()
        else:
            delta, clean_logits, grad = fgsm(model, img, lbl, **attack_params, device=device)
        # Make delta require gradients
        delta.requires_grad_(True)

        adv_logits = model(img + delta)
        optimizer.zero_grad()
        classification_loss = criterion(adv_logits, lbl)
        pairing_loss = F.mse_loss(adv_logits, clean_logits)
        loss = lambda1 * classification_loss + lambda2 * pairing_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Loss 
        batch_losses.append(loss.item())
        # Accuracy
        batch_correct = sum(torch.argmax(adv_logits, axis=1) == lbl).item()
        batch_size = lbl.shape[0]
        correct += batch_correct
        total += batch_size
        batch_accuracies.append(100 * batch_correct / batch_size)
        # Alignment
        grad_delta = delta.grad.clone().detach()
        cosine_similarity = F.cosine_similarity(grad.view(grad.size(0), -1), grad_delta.view(grad_delta.size(0), -1)).mean().item()
        batch_alignments.append(cosine_similarity)
    epcoh_accuracy = 100 * correct / total
    return batch_losses, batch_accuracies, batch_alignments, epcoh_accuracy


def train(architecture, loader, attack_params: dict, lambda1: float = 1.0, lambda2: float = 4.0, eval_mode_attack: bool = False, num_classes: int = 10, seed: int = 0, n_epochs: int = 30, device: str = 'cuda'):
    # Setting the seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = architecture(num_classes=num_classes).to(device)

    # Create saving directory
    saving_dir = f'experiments/{loader.dataset.__class__.__name__}_{model.__class__.__name__}'
    if not attack_params['clip']:
        saving_dir += '_N-ALP_'
    elif attack_params['k'] != 0:
        saving_dir += '_RS-ALP_'
    else:
        saving_dir += '_ALP_'
    saving_dir += f'E{n_epochs}_S{seed}_α={attack_params["alpha"]}_k={attack_params["k"]}_λ1={lambda1}_λ2={lambda2}'
    if eval_mode_attack:
        saving_dir += '_EMA'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Setup optimizer and scheduler 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_steps = n_epochs * len(loader) / 2
    scheduler = CyclicLR(optimizer, base_lr=0., max_lr=0.2, step_size_up=lr_steps, step_size_down=lr_steps)  
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    losses, accuracies, alignments = [], [], []
    torch.save(model.state_dict(), f'{saving_dir}/weights_00.pth')

    for epoch in range(n_epochs):
        batch_losses, batch_accuracies, batch_alignments, epcoh_accuracy = train_epoch(model, optimizer, scheduler, loader, criterion, attack_params, lambda1, lambda2, eval_mode_attack, device)
        print(f'Epcoh {str(epoch+1).zfill(2)} | Accuracy = {epcoh_accuracy:.2f}%')
        losses += batch_losses
        accuracies += batch_accuracies
        alignments += batch_alignments
        torch.save(model.state_dict(), f'{saving_dir}/weights_{str(epoch+1).zfill(2)}.pth')

    losses, accuracies, alignments = np.array(losses), np.array(accuracies), np.array(alignments)
    np.save(f'{saving_dir}/losses.npy', losses)
    np.save(f'{saving_dir}/accuracies.npy', accuracies)
    np.save(f'{saving_dir}/alignments.npy', alignments)
    return saving_dir
