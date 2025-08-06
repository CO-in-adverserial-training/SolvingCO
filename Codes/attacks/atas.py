import torch
import torch.nn.functional as F


def atas(model, x, y, index, upper_limit, lower_limit, mu, std, epsilon: float = 8/255,
          beta: float=0.5, gamma_over_c: float=16/255, c: float=0.01, warm_up_epoch: int=5,
          delta = None, moving_grad_norm = None, warm_up: bool=False):     
    model_training = model.training
    model.eval()
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
                  
    # Initialize random step
    if index is not None:
        delta = delta[index].clone().detach()
        moving_grad_norm = moving_grad_norm[index].clone().detach()
    else:
        delta = torch.empty_like(x).uniform_(-1, 1) * eps
        moving_grad_norm = 0
    x_adv = x + delta
    x_adv.requires_grad_()
    preds = model(x_adv)
    loss = F.cross_entropy(preds, y)
    grad = torch.autograd.grad(loss, x_adv)[0].detach()
    
    if not warm_up:
        with torch.no_grad():
            grad_norm = torch.norm(grad.view(len(grad), -1), dim=1).detach() ** 2
            moving_grad_norm = beta * moving_grad_norm + (1 - beta) * grad_norm
            step_size = gamma_over_c / (1 + torch.sqrt(moving_grad_norm) / c)
        step_size = step_size.view(-1, 1, 1, 1).expand_as(grad)
    else:
        step_size = eps
    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
    x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    x_adv = torch.clamp(x_adv, min=lower_limit, max=upper_limit)
    delta = (x_adv - x).detach()
    if model_training:
        model.train()
    
    step_size = step_size.mean().clone().detach()
    return delta, grad, moving_grad_norm, step_size