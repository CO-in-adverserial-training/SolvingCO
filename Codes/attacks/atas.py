import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR


def fgsm(model, x, y, gdnorm, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 1.0, k: float = 1.0, 
         clip: bool = False, beta: float = 0.5, min_step_size: float = 0.5, max_step_size: float = 1.75, c: float = 0.01, device: str = 'cuda'):     
    # Normalize perturbations
    epsilon = (epsilon / std).view(1, -1, 1, 1)
                  
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

    with torch.no_grad():
        cur_gdnorm = torch.norm(grad.view(x.size(0), -1), dim=1).detach() ** 2 * (1 - beta) + gdnorm * beta
        step_sizes = 1 / (1 + torch.sqrt(cur_gdnorm) / c) * alpha * epsilon
        if clip:
            step_sizes = torch.clamp(step_sizes, min_step_size * epsilon, max_step_size * epsilon)
    
    # Compute perturbation based on sign of gradient
    step_sizes = step_sizes.view(-1, 3, 1, 1).expand_as(grad)
    delta = eta + step_sizes * torch.sign(grad.detach())
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    if clip:
        delta = torch.clamp(delta, -epsilon, +epsilon)
    delta = delta.detach()
    
    return delta, grad, torch.mean(cur_gdnorm)
