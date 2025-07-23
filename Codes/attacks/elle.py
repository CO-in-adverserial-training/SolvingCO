import torch
import torch.nn.functional as F

#Implement ELLE Regularizer
def elle(x, y, model, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 8/255, k: float = 1.0):
    model_training = model.training
    model.eval()
    
    etaa = torch.empty_like(x).uniform_(-k, k)
    etaa *= epsilon.view(1, -1, 1, 1)  # Reshape epsilon for broadcasting
    etaa = torch.clamp(etaa, lower_limit - x, upper_limit - x)
    xa = x + etaa
    
    etab = torch.empty_like(x).uniform_(-k, k)
    etab *= epsilon.view(1, -1, 1, 1)  # Reshape epsilon for broadcasting
    etab = torch.clamp(etab, lower_limit - x, upper_limit - x)
    xb = x + etab
    
    alpha_p = torch.rand(1)[0]
    xc = (1 - alpha_p) * xa + alpha_p * xb
    preds_a = model(xa)
    cost_a = F.cross_entropy(preds_a, y)
    preds_b = model(xb)
    cost_b = F.cross_entropy(preds_b, y)
    preds_c = model(xc)
    cost_c = F.cross_entropy(preds_c, y)
    e_lin = torch.pow(cost_c.item() - (1 - alpha_p) * cost_a.item() - alpha_p * cost_b.item() , 2)
    e_lin = e_lin.detach()
    if model_training:
        model.train()
    return e_lin