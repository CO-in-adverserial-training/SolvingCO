import torch
import torch.nn.functional as F

def fastbat(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255):
    batch_size = x.size(0)
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    # Initialize random step
    eta = torch.empty_like(x).uniform_(-1, 1) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True

    output = model(x + eta)
    loss = -F.cross_entropy(output, y, reduction="sum")

    grad = torch.autograd.grad(loss, eta, create_graph=True, retain_graph=True)[0]
    
    delta = eta - alpha * grad
    delta = torch.clamp(delta, -eps, +eps)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    z = delta.clone().detach().view(batch_size, -1)

    z_min = torch.max(-x.view(batch_size, -1), -eps * torch.ones_like(x.view(batch_size, -1)))
    z_max = torch.min(1 - x.view(x.size(0), -1), eps * torch.ones_like(x.view(batch_size, -1)))
    H = ((z > z_min + 1e-7) & (z < z_max - 1e-7)).to(torch.float32)

    delta_cur = delta.detach().requires_grad_(True)
    model.zero_grad()
    
