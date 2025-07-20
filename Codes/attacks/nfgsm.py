import torch
import torch.nn.functional as F


def nfgsm(model, x, y, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 8/255, k: float = 2.0):
    # Initialize random step
    eta = torch.empty_like(x).uniform_(-k, k)
    eta *= epsilon.view(1, -1, 1, 1)  # Reshape epsilon for broadcasting
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True
    
    output = model(x + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0]
    grad = grad.detach()
    
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * torch.sign(grad)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad