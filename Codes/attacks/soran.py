import torch
import torch.nn.functional as F
from training.linearity_coef import calc_linearity_coef

def soran(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float= 8/255, max_alpha: float=4 / 255, max_perturb: float=16 / 255, method: str="Second Order Theory Sign", attack_iters: int=10):
    model_training = model.training
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    # Save maximum loss 
    max_loss = None
    delta_max_loss = None
    alpha_scalar_max_loss = None
    
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
            if max_loss is None or loss.item() > max_loss:
                max_loss = loss.item()
                delta_max_loss = eta.clone()
                alpha_scalar_max_loss = max_alpha
            grad = torch.autograd.grad(loss, eta)[0]
            grad = grad.detach()
            
            # Compute perturbation based on sign of gradient
            # interpolation_coeff = torch.rand_like(grad).float()
            interpolation_coeff = torch.ones_like(grad).float()
            alpha_scalar = sora_max_alpha_function(method, max_alpha, linearity_coef=1)
            alpha = (alpha_scalar / std).view(1, -1, 1, 1)
            delta = eta + alpha * interpolation_coeff * grad.sign()
            delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
            # delta = delta.detach()
            prev_grad = grad.clone()
        else:
            output = model(x + delta)
            loss = F.cross_entropy(output, y)
            if max_loss is None or loss.item() > max_loss:
                max_loss = loss.item()
                delta_max_loss = delta.clone()
                alpha_scalar_max_loss = alpha_scalar
            grad = torch.autograd.grad(loss, delta)[0]
            grad = grad.detach()
            
            # Compute perturbation based on sign of gradient
            # interpolation_coeff = torch.rand_like(grad).float()
            interpolation_coeff = torch.ones_like(grad).float()
            linearity_coef = calc_linearity_coef(prev_grad, grad, "Second Order Theory Sign")
            print(linearity_coef)
            linearity_coef = min(1, linearity_coef)
            alpha_scalar = sora_max_alpha_function(method, max_alpha, linearity_coef=linearity_coef)
            alpha = (alpha_scalar / std).view(1, -1, 1, 1)
            delta = delta + alpha * interpolation_coeff * grad.sign()
            delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
            # delta = delta.detach()
            prev_grad = grad.clone()
        delta = torch.clamp(delta, -max_perturb, max_perturb)

    output = model(x + delta)
    loss = F.cross_entropy(output, y)
    if max_loss is None or loss.item() > max_loss:
        max_loss = loss.item()
        delta_max_loss = delta.clone()
        alpha_scalar_max_loss = alpha_scalar
    
    delta_max_loss = delta_max_loss.detach()
    
    if model_training:
        model.train()
    return delta_max_loss, grad, alpha_scalar_max_loss

# Function For Mapping Alignment To Max Noise For SORA Method 
def sora_max_range_noise_function(func):
    match func:
        case "Fix":
            k = 1
    return k

# Function For Mapping Alignment To Max Alpha For SORA Method 
def sora_max_alpha_function(func, max_alpha, linearity_coef=None):
    print(linearity_coef)
    linearity_coef = 0 if linearity_coef is None else linearity_coef
    if func in ["Second Order Theory", "Second Order Theory Sign"]:
        if linearity_coef == 1:
            coef = 1
        else:
            coef = min(1, 0.1 / (1 - linearity_coef))
    alpha = coef * max_alpha
    return alpha