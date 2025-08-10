import torch
import torch.nn.functional as F
import numpy as np

def sia(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float= 8/255, max_alpha: float=16 / 255, method: str="Second Order Theory Sign", alignment: float=1, prev_batch_alpha: float=None, linearity_coef: float=None):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha_scalar = sia_max_alpha_function(method, alignment, max_alpha, a=0.1, b=5, prev_batch_alpha=prev_batch_alpha, linearity_coef=linearity_coef)
    alpha = (alpha_scalar / std).view(1, -1, 1, 1)
    
    # Initialize random step
    k = sia_max_range_noise_function("Fix", alignment, a=2, b=1.5)
    eta = torch.empty_like(x).uniform_(-k, k)
    eta *= eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True

    output = model(x + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0]
    grad = grad.detach()
    
    # Compute perturbation based on sign of gradient
    interpolation_coeff = torch.rand_like(grad).float()
    delta = eta + alpha * interpolation_coeff * grad.sign()
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad, alpha_scalar

# Function For Mapping Alignment To Max Noise For SIA Method 
def sia_max_range_noise_function(func, alignment, a=2, b=1.5):
    match func:
        case "Fix":
            k = 1
        case "Inverse":
            k = min(2, 1 / (1.5 * abs(alignment)))
    return k

# Function For Mapping Alignment To Max Alpha For SIA Method 
def sia_max_alpha_function(func, alignment, max_alpha, a=0.1, b=5, moving_avg_alignment=1, prev_batch_alpha=None, linearity_coef=None):
    alignment = 1 if alignment is None else alignment
    prev_batch_alpha = max_alpha if prev_batch_alpha is None else prev_batch_alpha
    linearity_coef = 0 if linearity_coef is None else linearity_coef
    match func:
        case "Linear": # Default a = 0.1, b = 5
            coef = min(1, max(a, b * alignment))
        case "Exponential Moving Avg": # Default a = 0.1, b = 1.5
            theta = 0.25 if alignment >= moving_avg_alignment else 0.05
            moving_avg_alignment = (1 - theta) * moving_avg_alignment + theta * alignment
            coef = min(1, max(a, b * moving_avg_alignment))
        case "Sigmoid": # Default a = 0.1, b = 5
            coef = a + (1 - a) / (1 + np.exp(-b * alignment - 0.2))
        case "Prev Batch Update":
            prev_batch_coef = prev_batch_alpha / max_alpha
            if alignment < 0.2:
                coef = max(0.1, 0.999 * prev_batch_coef)
            elif alignment > 0.4:
                coef = min(1, (100 / 95) * prev_batch_coef)
            else:
                coef = prev_batch_coef
        case func if func in ["Second Order Theory", "Second Order Theory Sign"]:
            if linearity_coef == 1:
                coef = 1
            else:
                coef = min(1, abs(0.1 / (1 - linearity_coef)))
    alpha = coef * max_alpha
    return alpha
