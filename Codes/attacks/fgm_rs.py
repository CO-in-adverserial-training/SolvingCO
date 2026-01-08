import torch
import torch.nn.functional as F


def fgm_rs(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 1.0, alpha: float = 1.25, k: float = 1.0):
    """
    Fast Gradient Method with Random Start (FGM).

    FGM-RS is a one-step adversarial attack used for adversarial training, where the
    perturbation is initialized with a random step within the ε-ball before applying
    the standard FGM update based on the loss gradient. This random initialization 
    improves robustness training over deterministic FGM by avoiding gradient masking.

    Reference:

    Args:
        model (torch.nn.Module): The target network.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized maximum limit for inputs.
        lower_limit (torch.Tensor): Per-channel normalized minimum limit for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 1.0).
        alpha (float): Gradient step size for perturbation (default: 1.25).
        k (float): Randomization scale factor for initial perturbation (default: 1.0).

    Returns:
        tuple:
            - torch.Tensor: Final FGM perturbation (`delta`).
            - torch.Tensor: Gradient tensor from backward pass.
    """
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    # Initialize random step
    eta = torch.randn_like(x) * k * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True

    # Compute gradient 
    output = model(x + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0].detach()

    # Compute perturbation based on gradient
    grad_normalized = grad / (grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-10)
    delta = alpha * grad_normalized
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad
