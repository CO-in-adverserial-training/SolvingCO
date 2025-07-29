import torch
import torch.nn.functional as F


def fgsm_rs(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 10/255, k: float = 1.0):
    # Initialize random step
    eta = torch.empty_like(x).uniform_(-k, k)
    eta *= epsilon.view(1, -1, 1, 1)  # Reshape epsilon for broadcasting
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True
    
    output = model(x + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0].detach()

    eps = (epsilon / std).view(1, 3, 1, 1)
    alpha = (alpha / std).view(1, 3, 1, 1)

    
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * torch.sign(grad)
    delta = torch.clamp(delta, -eps, +eps)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad