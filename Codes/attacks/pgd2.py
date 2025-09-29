import torch
import torch.nn.functional as F

def pgd2(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 4/255, attack_iters: int = 2, k: float = 1.0, clip: bool = True):
    """
    Projected Gradient Descent (PGD2) Adversarial Attack.

    PGD2 is an iterative extension of FGSM that applies multiple small-magnitude
    gradient sign steps, projecting the perturbations back into the allowed ε-ball
    after each update. This method is considered one of the strongest first-order
    adversaries and is widely used as a benchmark for adversarial robustness.

    Random initialization within the ε-ball (scaled by `k`) increases attack
    effectiveness by avoiding deterministic local optima. When `clip=True`, the
    perturbations are additionally clamped to the exact ε-bound per iteration.

    Reference:
        Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2019).
        "Towards Deep Learning Models Resistant to Adversarial Attacks."
        arXiv:1706.06083 (https://arxiv.org/abs/1706.06083)

    Args:
        model (torch.nn.Module): Target neural network.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized upper bound for inputs.
        lower_limit (torch.Tensor): Per-channel normalized lower bound for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        alpha (float): Step size for each PGD iteration (default: 2/255).
        attack_iters (int): Number of PGD iterations (default: 10).
        k (float): Scale factor for the random initialization (default: 1.0).
        clip (bool): Whether to clip perturbations to the ε-ball after each step
            (default: True).

    Returns:
        tuple:
            - torch.Tensor: Final adversarial perturbation (`delta`).
            - torch.Tensor: Gradient tensor from the last backward pass.
    """
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
