import torch
import torch.nn.functional as F


def fgsm(model, x, y, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 16/255):
    x.requires_grad = True
    output = model(x)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach()
    
    # Compute perturbation based on sign of gradient
    delta = alpha * torch.sign(grad)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = torch.clamp(delta, -epsilon, +epsilon)
    delta = delta.detach()
    
    return delta, grad