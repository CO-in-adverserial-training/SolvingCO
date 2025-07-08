import torch
import torch.nn.functional as F

def attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, attack_iters=10, restarts=1, norm="l_inf", early_stop=False, fgsm_init=None, device: str = 'cuda'):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if attack_iters>1 or fgsm_init=='random': 
            if norm == "l_inf":
                delta.uniform_(-epsilon.max(), epsilon.max())
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = torch.clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = torch.clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
            max_delta[all_loss >= max_loss] = torch.clone(delta.detach()[all_loss >= max_loss])
            max_loss = torch.max(max_loss, all_loss)
    return max_delta


def create_attack(eps, alpha_coef, lower_limit, upper_limit, attack_iters, random_start, std):
    epsilon = eps / std
    alpha = epsilon * alpha_coef
    def attack(model, x, y):
        fgsm_init = 'random' if random_start else None
        return attack_pgd(model, x, y, epsilon, alpha, lower_limit, upper_limit, attack_iters, fgsm_init=fgsm_init)
    return attack