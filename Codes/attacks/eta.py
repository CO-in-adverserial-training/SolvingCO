import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR


def fgsm_eta(model, x, y_relaxed, epsilon, alpha, k=2.0, clip=False,
             lower_limit=0.0, upper_limit=1.0, eta_momentum=0.75,
             prev_delta=None, device='cuda'):

    # Initialize momentum-based delta
    delta = torch.zeros_like(x).to(device)
    if prev_delta is None:
        for j in range(len(epsilon)):
            delta[:, j, :, :].uniform_(-k * epsilon[j][0][0].item(), k * epsilon[j][0][0].item())
        delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    else:
        delta = eta_momentum * prev_delta

    delta.requires_grad = True
    output = model(x + delta)
    loss = F.cross_entropy(output, y_relaxed)
    grad = torch.autograd.grad(loss, delta)[0].detach()

    # Standard FGSM update
    delta = delta + alpha * epsilon * torch.sign(grad)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    if clip:
        delta = torch.clamp(delta, -epsilon, +epsilon)

    return delta.detach(), grad


def dynamic_label_relaxation(y, epoch, total_epochs, beta=0.9, gamma_min=0.1, num_classes=10):
    delta = epoch / total_epochs
    gamma = max(beta * np.tanh(1 - delta), gamma_min)
    one_hot = F.one_hot(y, num_classes=num_classes).float()
    relaxed = one_hot * gamma + (1 - one_hot) * (1 - gamma) / (num_classes - 1)
    return relaxed


def taxonomy_loss(model, x_clean, x_adv, y_relaxed, lambda_reg=0.75):
    output_adv = model(x_adv)
    output_clean = model(x_clean).detach()
    ce = F.cross_entropy(output_adv, y_relaxed, reduction='none')
    p = torch.gather(F.softmax(output_adv, dim=1), 1, y_relaxed.argmax(dim=1, keepdim=True)).squeeze(1)
    reg_term = lambda_reg * (F.mse_loss(output_adv, output_clean, reduction='none').mean(dim=1)) * torch.tanh(1 - p)
    return ce + reg_term


def train_epoch_eta(model, optimizer, scheduler, loader, criterion, attack_params,
                    epoch, n_epochs, prev_deltas, device='cuda'):
    batch_losses, batch_accuracies, batch_alignments = [], [], []
    total, correct = 0, 0

    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_relaxed = dynamic_label_relaxation(y, epoch, n_epochs,
                                             num_classes=attack_params['num_classes']).to(device)

        delta, grad = fgsm_eta(
            model, x, y_relaxed,
            epsilon=attack_params['epsilon'], alpha=attack_params['alpha'],
            k=attack_params['k'], clip=attack_params['clip'],
            lower_limit=attack_params['lower_limit'], upper_limit=attack_params['upper_limit'],
            eta_momentum=attack_params['eta_momentum'],
            prev_delta=prev_deltas[i] if prev_deltas is not None else None,
            device=device
        )

        x_adv = torch.clamp(x + delta, attack_params['lower_limit'], attack_params['upper_limit'])
        x_adv.requires_grad = True

        # Taxonomy Loss
        loss_vec = taxonomy_loss(model, x, x_adv, y_relaxed, lambda_reg=attack_params['lambda'])

        # COLA
        pred_clean = model(x).argmax(dim=1)
        scale = torch.where(pred_clean == y, torch.tensor(attack_params['cola_eta'], device=device), 1.0)
        loss = (loss_vec * scale).mean()

        delta.requires_grad_()
        optimizer.zero_grad()
        pred_adv = model(x + delta)
        loss = criterion(pred_adv, y)  # dummy loss to ensure delta.grad exists
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # Accuracy
        pred = pred_adv
        batch_correct = (pred.argmax(dim=1) == y).sum().item()
        correct += batch_correct
        total += y.size(0)
        batch_accuracies.append(100 * batch_correct / y.size(0))
        batch_losses.append(loss.item())

        # Cosine Alignment
        grad_delta = delta.grad.clone().detach()
        cos_sim = F.cosine_similarity(grad.view(grad.size(0), -1),
                                      grad_delta.view(grad_delta.size(0), -1)).mean().item()
        batch_alignments.append(cos_sim)

        # Save delta for momentum
        if prev_deltas is not None:
            with torch.no_grad():
                prev_deltas[i] = delta.detach()

    epoch_accuracy = 100 * correct / total
    return batch_losses, batch_accuracies, batch_alignments, epoch_accuracy, prev_deltas


def train(architecture, loader, attack_params, num_classes=10, seed=0, n_epochs=30, device='cuda'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = architecture(num_classes=num_classes).to(device)

    saving_dir = f'experiments/{loader.dataset.__class__.__name__}_{model.__class__.__name__}_ETA_E{n_epochs}_S{seed}_α{attack_params["alpha"]}_k{attack_params["k"]}_η{attack_params["eta_momentum"]}_λ{attack_params["lambda"]}'
    os.makedirs(saving_dir, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_steps = n_epochs * len(loader) / 2
    scheduler = CyclicLR(optimizer, base_lr=0., max_lr=0.2, step_size_up=lr_steps, step_size_down=lr_steps)
    criterion = nn.CrossEntropyLoss()

    torch.save(model.state_dict(), f'{saving_dir}/weights_00.pth')

    losses, accuracies, alignments = [], [], []
    prev_deltas = [None] * len(loader)

    for epoch in range(n_epochs):
        batch_losses, batch_accuracies, batch_alignments, acc, prev_deltas = train_epoch_eta(
            model, optimizer, scheduler, loader, criterion, attack_params,
            epoch, n_epochs, prev_deltas, device
        )
        print(f"Epoch {epoch+1:02d} | Accuracy = {acc:.2f}%")
        losses += batch_losses
        accuracies += batch_accuracies
        alignments += batch_alignments
        torch.save(model.state_dict(), f'{saving_dir}/weights_{str(epoch+1).zfill(2)}.pth')

    np.save(f'{saving_dir}/losses.npy', np.array(losses))
    np.save(f'{saving_dir}/accuracies.npy', np.array(accuracies))
    np.save(f'{saving_dir}/alignments.npy', np.array(alignments))
    return saving_dir