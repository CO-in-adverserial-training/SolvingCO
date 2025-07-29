import torch
import torch.nn.functional as F

def pgd(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 2/255, attack_iters: int = 10, k: float = 1.0, clip: bool = True):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    # Initialize random step
    delta = torch.empty_like(x).uniform_(-k, k) * eps
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x).detach()
    delta.requires_grad = True

    for _ in range(attack_iters):
        output = model(x + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        with torch.no_grad():
            delta.data += alpha * torch.sign(grad)
            if clip:
                delta.data.clamp_(-eps, eps)
            delta.data.clamp_(lower_limit - x, upper_limit - x)
        delta.grad.zero_()
    delta = delta.detach()

    return delta, grad