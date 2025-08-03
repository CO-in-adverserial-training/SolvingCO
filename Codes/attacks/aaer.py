import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR


def l2_square(x,y):
    diff = x-y
    diff = diff*diff
    diff = diff.sum(1).mean(0)
    return diff


def fgsm(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 2.0, clip: bool = False, device: str = 'cuda'):
    # Normalize perturbations
    epsilon = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    # Initialize random step
    eta = torch.zeros_like(x).to(device)
    if k != 0:
        for j in range(len(epsilon)):
            eta[:, j, :, :].uniform_(-k * epsilon[j][0][0].item(), k * epsilon[j][0][0].item())
        eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True
    
    output = model(x + eta)
    output_org = output.detach()
    loss = nn.CrossEntropyLoss(reduce=False)(output, y)
    loss_before = loss.detach()
    loss = loss.mean()
    loss.backward()
    grad = eta.grad.detach()

    delta = eta + alpha * torch.sign(grad)
    if clip:
        delta = torch.clamp(delta, -epsilon, +epsilon)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad, output_org, loss_before


def train_epoch(model, optimizer, scheduler, loader, attack_params, lambda1: float = 1.0, lambda2: float = 4.0, lambda3: float = 1.5, eval_mode_attack: bool = False, device: str = 'cuda'):
    batch_losses = []
    batch_accuracies = []
    batch_alignments = []
    total, correct = 0, 0
    ae_num, ae_ce_loss, ae_l2_loss = 0, 0, 0

    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        # Attack
        if eval_mode_attack:
            model.eval()
            delta, grad, output_org, loss_before = fgsm(model, img, lbl, **attack_params, device=device)
            model.train()
        else:
            delta, grad, output_org, loss_before = fgsm(model, img, lbl, **attack_params, device=device)
        delta.requires_grad_(True)

        pred = model(img + delta)
        loss = nn.CrossEntropyLoss(reduce=False)(pred, lbl)
        loss_after = loss.detach()
        loss = loss.mean()

        abnormal_example = loss_before > loss_after
        normal_example = loss_before <= loss_after
        abnormal_count = torch.count_nonzero(abnormal_example)
        normal_count = torch.count_nonzero(normal_example)

        if abnormal_count != 0:
            abnormal_variation = l2_square(output_org[abnormal_example], pred[abnormal_example])
            abnormal_ce = abnormal_example * (loss_before - loss_after)
            abnormal_ce = abnormal_ce.sum() / abnormal_count
            ae_num = ae_num + abnormal_count

        if normal_count != 0:
            normal_variation = l2_square(output_org[normal_example], pred[normal_example])

        if abnormal_count != 0 and normal_count != 0:
            loss = loss + (lambda1 * abnormal_count / lbl.size(0)) * (lambda2 * abnormal_ce + lambda3 * max(abnormal_variation - normal_variation.item(), 0)) # '* min((epoch/20), 1)' warm-up for long training schedule
            ae_ce_loss = ae_ce_loss + abnormal_ce.item() * abnormal_count
            ae_l2_loss = ae_l2_loss + (abnormal_variation.item() - normal_variation.item()) * abnormal_count

        optimizer.zero_grad()
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


def train(architecture, loader, attack_params: dict, lambda1: float = 1.0, lambda2: float = 4.0, lambda3: float = 1.5, seed: int = 0, n_epochs: int = 30, eval_mode_attack: bool = False, device: str = 'cuda'):
    # Setting the seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = architecture().to(device)

    # Create saving directory
    saving_dir = f'experiments/{loader.dataset.__class__.__name__}_{model.__class__.__name__}'
    if not attack_params['clip']:
        saving_dir += '_N-AAER_'
    elif attack_params['k'] != 0:
        saving_dir += '_RS-AAER_'
    else:
        saving_dir += '_AAER_'
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
    
    losses, accuracies, alignments = [], [], []
    torch.save(model.state_dict(), f'{saving_dir}/weights_00.pth')

    for epoch in range(n_epochs):
        batch_losses, batch_accuracies, batch_alignments, epcoh_accuracy = train_epoch(model, optimizer, scheduler, loader, attack_params, lambda1, lambda2, lambda3, eval_mode_attack, device)
        print(f'Epcoh {str(epoch+1).zfill(2)}: Accuracy = {epcoh_accuracy:.2f}%')
        losses += batch_losses
        accuracies += batch_accuracies
        alignments += batch_alignments
        torch.save(model.state_dict(), f'{saving_dir}/weights_{str(epoch+1).zfill(2)}.pth')

    losses, accuracies, alignments = np.array(losses), np.array(accuracies), np.array(alignments)
    np.save(f'{saving_dir}/losses.npy', losses)
    np.save(f'{saving_dir}/accuracies.npy', accuracies)
    np.save(f'{saving_dir}/alignments.npy', alignments)
    return saving_dir
