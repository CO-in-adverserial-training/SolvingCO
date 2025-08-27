import torch
import torch.nn.functional as F
import copy

#Implement ELLE Regularizer
def elle(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 1.0):
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    # Generate adversarial example using FGSM-RS
    # Initialize random step
    eta = torch.empty_like(x).uniform_(-k, k) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    x_adv = copy.deepcopy(x) + eta

    x_adv.requires_grad=True
    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, y)
    loss.backward(retain_graph=True)
    grads_input = copy.deepcopy(x_adv.grad)
    grad = x_adv.grad.detach()
    
    x_adv = x_adv + alpha * torch.sign(grads_input)
    x_adv = torch.clamp(x_adv, x - eps, x + eps)
    x_adv = torch.clamp(x_adv, lower_limit, upper_limit)
    model.zero_grad()

    x_adv.detach()

    out2 = model(x_adv)
    loss = F.cross_entropy(out2,y)

    bs = x.shape[0]
    x_ab = x.repeat([2,1,1,1]) 
    etaab = torch.empty_like(x_ab).uniform_(-k, k) * eps
    etaab = torch.clamp(etaab, lower_limit - x_ab, upper_limit - x_ab)
    x_ab = x_ab + etaab
    alphaa = torch.rand([bs,1,1,1],device = x.device)
    x_c = (1-alphaa)*x_ab[:bs] + alphaa*x_ab[bs:]
    alphaa = alphaa.squeeze()

    # Forward pass
    losses = F.cross_entropy(model(torch.cat((x_ab,x_c),dim=0)), y.repeat([3]), reduction='none')

    # Regularization term
    lin_err = F.mse_loss(losses[2*bs:], (1-alphaa)*losses[:bs] + alphaa*losses[bs:2*bs])

    delta = (x_adv - x).detach()
    return delta, lin_err, grad