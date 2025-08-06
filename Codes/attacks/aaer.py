import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def l2_square(x,y):
    diff = x - y
    diff = diff * diff
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
    
    return delta, grad, clean_logit, loss_before


def aaer(loss_before, loss_after, clean_logit, adv_logit, lambda1: float = 1.0, lambda2: float = 4.0, lambda3: float = 1.5):
    abnormal_example = loss_before > loss_after
    normal_example = loss_before <= loss_after
    abnormal_count = torch.count_nonzero(abnormal_example)
    normal_count = torch.count_nonzero(normal_example)
    total_count = abnormal_count + normal_count

    # AAE-CE and AAE-L2
    if abnormal_count != 0:
        abnormal_variation = l2_square(clean_logit[abnormal_example], adv_logit[abnormal_example])
        abnormal_ce = abnormal_example * (loss_before - loss_after)
        abnormal_ce = abnormal_ce.sum() / abnormal_count
    # NAE-L2
    if normal_count != 0:
        normal_variation = l2_square(clean_logit[normal_example], adv_logit[normal_example])
    # AAER
    if abnormal_count != 0 and normal_count != 0:
        constrained_variation = max(abnormal_variation - normal_variation.item(), 0)
        reg_loss = (lambda1 * abnormal_count / total_count) * (lambda2 * abnormal_ce + lambda3 * constrained_variation) # '* min((epoch/20), 1)' warm-up for long training schedule
    
    return reg_loss


