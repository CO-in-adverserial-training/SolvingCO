import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Theoretically Principled Trade-off between Robustness and Accuracy(TRADES)
def trades(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, perturb_steps: int = 10, alpha: float= 2/255):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    batch_size = x.shape[0]
    kl_criterion = nn.KLDivLoss(reduction = "sum")
    model_training = model.training
    model.eval()
    delta = 0.001 * torch.randn(x.shape, device=x.device).detach()
    delta = delta * (1 / std).view(1, -1, 1, 1)

    x_trades = x + delta
    for step in range(perturb_steps):
        x_trades.requires_grad_(True)
        with torch.enable_grad():
            loss_kl = kl_criterion(F.log_softmax(model(x_trades), dim=1), F.softmax(model(x), dim=1))
        grad = torch.autograd.grad(loss_kl, x_trades)[0]
        if step == 0:
            grad_zero = grad.detach()
        x_trades = x_trades.detach() + alpha * torch.sign(grad.detach())
        x_trades = torch.clamp(x_trades, min=x - eps, max=x + eps)
        x_trades = torch.clamp(x_trades, min=lower_limit, max=upper_limit)
    if model_training:
        model.train()
    loss_robust = (1.0 / batch_size) * kl_criterion(F.log_softmax(model(x_trades), dim=1), F.softmax(model(x), dim=1))
    
    return torch.zeros_like(x), loss_robust, grad_zero
