import numpy as np 
import torch
import torch.nn as nn
from collections import OrderedDict

def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict

def add_into_weights(model, diff, gamma, beta, layer_number):
    names_in_diff = diff.keys()
    diff_count = 0
    with torch.no_grad():
        for i, (name, param) in enumerate(model.named_parameters()):
            if name in names_in_diff:
                diff_count += 1
                layer_weight = 1.0 - np.power(np.log(diff_count) / np.log(layer_number), gamma) #21 for resnet-18, 35 for wederesnet-34, 50 for Vit-small
                param.add_(layer_weight * beta * diff[name])


def lap(model, proxy, opt, proxy_opt, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 2.0, clip: bool = False, beta: float = 0.01, gamma: float 1.0, layer_number: int = 21):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    # Initialize random step
    eta = torch.empty_like(x).uniform_(-k, k) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True
    
    output = model(x + eta)
    loss = nn.CrossEntropyLoss(reduce=True)(output, y)

    proxy.load_state_dict(model.state_dict())
    proxy_opt.load_state_dict(opt.state_dict())

    opt.zero_grad()
    loss.backward()
    opt.step()
  
    grad = delta.grad.detach()
    
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * torch.sign(grad)
    delta = torch.clamp(delta, -eps, +eps)
    if clip:
        delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()

    diff_weights = diff_in_weights(proxy, model)
    model.load_state_dict(proxy.state_dict())
    opt.load_state_dict(proxy_opt.state_dict())

    add_into_weights(model, diff_weights, gamma, beta, layer_number)
    
    return delta, grad
