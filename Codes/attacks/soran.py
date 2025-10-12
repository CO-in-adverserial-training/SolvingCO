import torch
import torch.nn.functional as F
from training.linearity_coef import calc_linearity_coef

def soran(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float= 8/255, max_alpha: float=4 / 255, max_perturb: float=16 / 255, method: str="Second Order Theory Sign", attack_iters: int=10, linearity_coef: float=None):
    model_training = model.training
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    
    # Initialize random step
    k = sora_max_range_noise_function("Fix")
    eta = torch.empty_like(x).uniform_(-k, k)
    eta *= eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True

    for iteration in range(attack_iters):
        if iteration == 0:
            output = model(x + eta)
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, eta)[0]
            grad = grad.detach()
            
            # Compute perturbation based on sign of gradient
            interpolation_coeff = torch.rand_like(grad).float()
            alpha_scalar = sora_max_alpha_function(method, max_alpha, linearity_coef=linearity_coef)
            alpha = (alpha_scalar / std).view(1, -1, 1, 1)
            delta = eta + alpha * interpolation_coeff * grad.sign()
            delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
            # delta = delta.detach()
            prev_grad = grad.clone()
        else:
            output = model(x + delta)
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, delta)[0]
            grad = grad.detach()
            
            # Compute perturbation based on sign of gradient
            interpolation_coeff = torch.rand_like(grad).float()
            linearity_coef = min(1, calc_linearity_coef(prev_grad, grad, "Second Order Theory Sign"))
            alpha_scalar = sora_max_alpha_function(method, max_alpha, linearity_coef=linearity_coef)
            alpha = (alpha_scalar / std).view(1, -1, 1, 1)
            delta = delta + alpha * interpolation_coeff * grad.sign()
            delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
            # delta = delta.detach()
            prev_grad = grad.clone()
        delta = torch.clamp(delta, -max_perturb, max_perturb)
    
    delta = delta.detach()
    if model_training:
        model.train()
    return delta, grad, alpha_scalar

# Function For Mapping Alignment To Max Noise For SORA Method 
def sora_max_range_noise_function(func):
    match func:
        case "Fix":
            k = 1
    return k

# Function For Mapping Alignment To Max Alpha For SORA Method 
def sora_max_alpha_function(func, max_alpha, linearity_coef=None):
    linearity_coef = 0 if linearity_coef is None else linearity_coef
    if func in ["Second Order Theory", "Second Order Theory Sign"]:
        if linearity_coef == 1:
            coef = 1
        else:
            coef = min(1, 0.1 / (1 - linearity_coef))
    alpha = coef * max_alpha
    return alpha
