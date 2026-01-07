import torch
import torch.nn.functional as F


def fgm(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 0.5, alpha: float = 1.0, norm: str = "L2"):
    """
    Fast Gradient Method (FGM).

    FGM is a single-step adversarial attack that perturbs inputs in the direction of
    the loss gradient with respect to the input. This method is simple but effective, 
    and is widely used for both evaluating and training models to improve adversarial 
    robustness.

    Reference:

    Args:
        model (torch.nn.Module): The target network.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized maximum limit for inputs.
        lower_limit (torch.Tensor): Per-channel normalized minimum limit for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 0.5).
        alpha (float): Gradient step size for perturbation (default: 1.0).

    Returns:
        tuple:
            - torch.Tensor: Final FGM perturbation (`delta`).
            - torch.Tensor: Gradient tensor from backward pass.
    """
    # Normalize perturbations
    if norm == "Linf":
        eps = (epsilon / std).view(1, -1, 1, 1)
        alpha = (alpha / std).view(1, -1, 1, 1)
    elif norm == "L2":
        eps = torch.sqrt(torch.sum((epsilon / std) ** 2)).item()
        alpha = torch.sqrt(torch.sum((alpha / std) ** 2)).item()

    x = x.clone().detach()
    x.requires_grad = True
    
    output = model(x)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, x)[0].detach()
    

    # Compute perturbation based on gradient
    if norm == "Linf":
        delta = alpha * torch.sign(grad)
        delta = torch.clamp(delta, -eps, +eps)
    elif norm == "L2":
        grad_normalized = grad / (grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-10)
        delta = alpha * grad_normalized
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad
