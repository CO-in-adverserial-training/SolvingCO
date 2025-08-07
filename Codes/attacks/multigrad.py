import torch
import torch.nn.functional as F


def multi_grad(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 10/255, samples: int = 3, zeroing_th: float = 1.0, k: float = 1.0, parallel: bool = True):
    if zeroing_th==-1:
        zeroing_th = samples
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    if parallel:
        x_cat = torch.cat([x for i in range(samples)], dim=0)
        # Initialize random step
        delta_cat = torch.empty_like(x_cat).uniform_(-k, k) * eps
        delta_cat = torch.clamp(delta_cat, lower_limit - x_cat, upper_limit - x_cat)
        delta_cat.requires_grad = True

        y_cat = torch.cat([y for i in range(samples)], dim=0)
        output = model(x_cat + delta_cat)
        F.cross_entropy(output, y_cat).backward()
        grad_cat = delta_cat.grad.detach()
        grads = [grad_cat[i*x.size(0):(i+1)*x.size(0)] for i in range(samples)]
    else:
        grads = []
        for _ in range(samples):
            # Initialize random step
            delta = torch.empty_like(x).uniform_(-k, k) * eps
            delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
            delta.requires_grad = True

            output = model(x + delta)
            F.cross_entropy(output, y).backward()

            grads += [torch.clone(delta.grad.detach())]
    g = sum([torch.sign(grads[i]) for i in range(samples)])
    grad = torch.where(torch.abs(g) < 
            (zeroing_th - (samples - zeroing_th)),
            torch.zeros_like(g), g)
    delta = torch.zeros_like(x).cuda() 
    d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
    d = torch.clamp(d, lower_limit - x, upper_limit - x)
    
    avg_grad = sum(grads).detach() / samples
    return d.detach(), avg_grad
