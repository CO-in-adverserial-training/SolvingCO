import torch
import torch.nn.functional as F

#Implementation of GradAlign Regularizer
def grad_align(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 1.0):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    x.requires_grad = True
    preds1 = model(x)
    cost1 = F.cross_entropy(preds1, y)
    grad1 = torch.autograd.grad(cost1, x)[0]
    grad1 = grad1.detach()
    eta = torch.empty_like(x).uniform_(-k, k) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    
    x_aug = x + eta
    preds2 = model(x_aug)
    cost2 = F.cross_entropy(preds2, y)
    grad2 = torch.autograd.grad(cost2, x)[0]
    grad2_copy = grad2.clone().detach()

    grad1 = grad1.reshape(grad1.shape[0], -1)
    grad2_copy = grad2_copy.reshape(grad2.shape[0], -1)   
    alignment = F.cosine_similarity(grad1, grad2_copy, dim=1)
    
    # Generate FGSM-RS Sample
    delta = eta + alpha * grad2.sign()
    delta = torch.clamp(delta, min=-eps, max=eps)
    delta = delta.detach()
    
    return delta, 1 - alignment.mean(), grad2
