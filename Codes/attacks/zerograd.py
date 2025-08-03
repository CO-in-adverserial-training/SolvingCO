import torch
import torch.nn.functional as F


def zero_grad(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 10/255, q_val: float = 0.35, k: float = 1.0, clip: bool = True):
    # Normalize perturbations
    epsilon = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    delta = torch.empty_like(x)
    if k != 0:
        for j in range(len(epsilon)):
            delta[:, j, :, :].uniform_(-k * epsilon[j][0][0].item(), k * epsilon[j][0][0].item())
        delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta.requires_grad = True

    output = model(x + delta)
    F.cross_entropy(output, y).backward()
    grad = delta.grad.detach()
    q_grad = torch.quantile(torch.abs(grad).view(grad.size(0), -1), q_val, dim=1)
    grad[torch.abs(grad) < q_grad.view(grad.size(0), 1, 1, 1)] = 0

    delta = delta + alpha * torch.sign(grad)
    if clip:
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x).detach()

    return delta, grad
