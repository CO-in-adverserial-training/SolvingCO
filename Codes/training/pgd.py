import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR


def pgd(model, x, y, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 0.25, attack_iters: int = 10, k: float = 1.0, clip: bool = False, device: str = 'cuda'):
    # Initialize random step
    delta = torch.zeros_like(x).to(device)
    if k != 0:
        for j in range(len(epsilon)):
            delta[:, j, :, :].uniform_(-k * epsilon[j][0][0].item(), k * epsilon[j][0][0].item())
        delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta.requires_grad = True

    for _ in range(attack_iters):
        output = model(x + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = delta + alpha * epsilon * torch.sign(grad)
        if clip:
            delta.data = torch.clamp(delta, -epsilon, epsilon)
        delta.data = torch.clamp(delta, lower_limit - x, upper_limit - x)
        delta.grad.zero_()
    delta = delta.detach()

    return delta, grad


def train_epoch(model, optimizer, scheduler, loader, criterion, attack_params, eval_mode_attack: bool, device: str = 'cuda'):
    batch_losses = []
    batch_accuracies = []
    batch_alignments = []
    total, correct = 0, 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        # Attack
        if eval_mode_attack:
            model.eval()
            delta, grad = pgd(model, img, lbl, **attack_params, device=device)
            model.train()
        else:
            delta, grad = pgd(model, img, lbl, **attack_params, device=device)
        # Make delta require gradients
        delta.requires_grad_(True)
        pred = model(img + delta)
        optimizer.zero_grad()
        loss = criterion(pred, lbl)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Loss 
        batch_losses.append(loss.item())
        # Accuracy
        batch_correct = sum(torch.argmax(pred, axis=1) == lbl).item()
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


def train(architecture, loader, attack_params: dict, eval_mode_attack: bool = False, num_classes: int = 10, seed: int = 0, n_epochs: int = 30, device: str = 'cuda'):
    # Setting the seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = architecture(num_classes=num_classes).to(device)

    # Create saving directory
    saving_dir = f'experiments/{loader.dataset.__class__.__name__}_{model.__class__.__name__}'
    if attack_params['clip']:
        saving_dir += f'_PGD{attack_params["attack_iters"]}_'
    else:
        saving_dir += f'_N-PGD{attack_params["attack_iters"]}_'
    saving_dir += f'E{n_epochs}_S{seed}_α{attack_params["alpha"]}_k{attack_params["k"]}'
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
        batch_losses, batch_accuracies, batch_alignments, epcoh_accuracy = train_epoch(model, optimizer, scheduler, loader, criterion, attack_params, eval_mode_attack, device)
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
