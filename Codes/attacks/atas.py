import torch
import torch.nn.functional as F
from training.utils import aug, aug_trans

def atas(model, x, y, index, upper_limit, lower_limit, mu, std, epsilon: float = 8/255,
          beta: float=0.5, gamma_over_c: float=16/255, c: float=0.01, min_step_size: float=4/255,
            max_step_size: float=14/255, warm_up_epoch: int=5, delta = None, moving_grad_norm = None, warm_up: bool=False):     
    model_training = model.training
    model.eval()
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
                  
    # Initialize random step
    if index is not None:
        delta_aug, transform_info = aug(delta[index].clone().detach())
        moving_grad_norm = moving_grad_norm[index].clone().detach()
    else:
        delta_aug = torch.empty_like(x).uniform_(-1, 1) * eps
        moving_grad_norm = torch.zeros(x.size(0), device=x.device)
        transform_info = None

    x = aug_trans(x, transform_info) if index is not None else x
    delta_aug.requires_grad_(True)
    preds = model(x + delta_aug)
    loss = F.cross_entropy(preds, y)
    grad = torch.autograd.grad(loss, delta_aug)[0].detach()
    
    if not warm_up:
        with torch.no_grad():
            grad_norm = torch.norm(grad.view(len(grad), -1), dim=1).detach() ** 2
            moving_grad_norm = beta * moving_grad_norm + (1 - beta) * grad_norm
            step_size = gamma_over_c / (1 + torch.sqrt(moving_grad_norm) / c)
            step_size = torch.clamp(step_size, min_step_size, max_step_size)
        # After computing per-sample scalar step_size (shape: [B])
        step_size = step_size.view(-1, 1, 1, 1)           # B × 1 × 1 × 1 (batch-specific scalar)
        step_size = step_size / std.view(1, -1, 1, 1)     # scale per channel (B × C × 1 × 1)
        step_size = step_size.expand_as(grad)             # B × C × H × W for broadcast in update

    else:
        step_size = eps
    delta_aug = delta_aug.detach() + step_size * torch.sign(grad.detach())
    delta_aug = torch.clamp(delta_aug, min=-eps, max=eps)
    delta_aug = torch.clamp(delta_aug, min=lower_limit - x, max=upper_limit - x)
    delta_aug = delta_aug.detach()
    # delta = inverse_aug(torch.zeros_like(delta), delta_aug, transform_info)

    if model_training:
        model.train()
    
    step_size = step_size.mean().item()
    return delta_aug, transform_info, grad, moving_grad_norm, step_size