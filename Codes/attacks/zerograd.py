import torch
import torch.nn.functional as F


def zero_grad(model, x, y, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 2.0, q_val: float = 0.35, k: float = 1.0, clip: bool = True):
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

    if clip:
        delta = torch.clamp(delta + alpha * epsilon * torch.sign(grad), min=-epsilon, max=epsilon)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x).detach()

    return delta, grad