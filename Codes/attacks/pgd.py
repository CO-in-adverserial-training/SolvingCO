import torch
import torch.nn.functional as F

def pgd(model, x, y, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 2/255, attack_iters: int = 10, k: float = 1.0, clip: bool = True):
    # Initialize random step
    delta = torch.empty_like(x).uniform_(-k, k)
    delta *= epsilon.view(1, -1, 1, 1)  # Reshape epsilon for broadcasting
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta.requires_grad = True

    for _ in range(attack_iters):
        output = model(x + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = delta + alpha * torch.sign(grad)
        if clip:
            delta.data = torch.clamp(delta, -epsilon, epsilon)
        delta.data = torch.clamp(delta, lower_limit - x, upper_limit - x)
        delta.grad.zero_()
    delta = delta.detach()

    return delta, grad