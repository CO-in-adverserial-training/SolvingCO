import torch
import torch.nn.functional as F
from .fgsm import fgsm

#Implement ELLE Regularizer
def elle(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 1.0):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    # FGSM-RS random points in the epsilon ball per sample
    etaa = torch.empty_like(x).uniform_(-k, k) * eps
    xa = x + etaa.clamp(lower_limit - x, upper_limit - x)

    etab = torch.empty_like(x).uniform_(-k, k) * eps
    xb = x + etab.clamp(lower_limit - x, upper_limit - x)

    # Random convex combination per sample
    alpha_p = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    xc = (1 - alpha_p) * xa + alpha_p * xb

    # Per-sample losses
    cost_a = F.cross_entropy(model(xa), y, reduction='none')
    cost_b = F.cross_entropy(model(xb), y, reduction='none')
    cost_c = F.cross_entropy(model(xc), y, reduction='none')

    # ELLE Error (mean squared error of "linearity")
    e_lin = ((cost_c - (1 - alpha_p.squeeze()) * cost_a - alpha_p.squeeze() * cost_b).pow(2)).mean()

    # Generate adversarial example using FGSM
    delta, grad = fgsm(model, x, y, upper_limit, lower_limit, epsilon, alpha)

    return delta, e_lin, grad