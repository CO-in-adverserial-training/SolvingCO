import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Theoretically Principled Trade-off between Robustness and Accuracy(TRADES)
def trades(x, y, model, upper_limit, lower_limit, epsilon: float = 8/255, perturb_steps: int=10, step_size: float=0.007):
    batch_size = x.shape[0]
    kl_criterion = nn.KLDivLoss(reduction = "sum")
    model_training = model.training
    model.eval()
    with torch.no_grad():
        clean_logits = model(x)
        clean_probs  = F.softmax(clean_logits, dim=1)
    delta = 0.001 * torch.randn(x.shape).detach()

    x_trades = x + delta
    for step in range(perturb_steps):
        x_trades.requires_grad_(True)
        with torch.enable_grad():
            loss_kl = kl_criterion(F.log_softmax(model(x_trades), dim=1), clean_probs)
        grad = torch.autograd.grad(loss_kl, x_trades)[0]
        if step == 0:
            grad = grad.detach()
        x_trades = x_trades.detach() + step_size * torch.sign(grad)
        x_trades = torch.clamp(x_trades, min=x - epsilon, max=x + epsilon)
        x_trades = torch.clamp(x_trades, min=lower_limit, max=upper_limit)
    if model_training:
        model.train()
    loss_robust = (1.0 / batch_size) * kl_criterion(F.log_softmax(model(x_trades), dim=1), clean_probs)
    
    return torch.zeros_like(x), loss_robust, grad