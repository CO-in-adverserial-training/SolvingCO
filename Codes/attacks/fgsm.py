import torch
import torch.nn.functional as F


def fgsm(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 16/255):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    x = x.clone().detach()
    x.requires_grad = True
    
    output = model(x)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, x)[0].detach()
    

    # Compute perturbation based on sign of gradient
    delta = alpha * torch.sign(grad)
    delta = torch.clamp(delta, -eps, +eps)
    torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad