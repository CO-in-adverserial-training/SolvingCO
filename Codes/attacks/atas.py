import torch
import torch.nn.functional as F
import random

def aug(input_tensor):
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size
    rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)
    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 8)
        y_t = random.randint(0, 8)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}


def aug_trans(input_tensor, transform_info):
    batch_size = input_tensor.shape[0]
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flip = transform_info['flipped']
    rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)

    for i in range(batch_size):
        flip_t = int(flip[i])
        x_t = int(x[i])
        y_t = int(y[i])
        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
    return rst


def inverse_aug(source_tensor, adv_tensor, transform_info):
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flipped = transform_info['flipped']
    batch_size = source_tensor.shape[0]

    for i in range(batch_size):
        flip_t = int(flipped[i])
        x_t = int(x[i])
        y_t = int(y[i])
        if flip_t:
            adv_tensor[i] = torch.flip(adv_tensor[i], [2])
        source_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32] = adv_tensor[i]

    return source_tensor

def atas(model, x, y, index, upper_limit, lower_limit, mu, std, epsilon: float = 8/255,
          beta: float=0.5, gamma_over_c: float=16/255, c: float=0.01, min_step_size: float=4/255,
            max_step_size: float=14/255, warm_up_epoch: int=5, delta = None, moving_grad_norm = None, warm_up: bool=False):     
    model_training = model.training
    model.eval()
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
                  
    # Initialize random step
    if index is not None:
        delta = delta[index].clone().detach()
        moving_grad_norm = moving_grad_norm[index].clone().detach()
    else:
        delta = torch.empty_like(x).uniform_(-1, 1) * eps
        moving_grad_norm = torch.zeros(x.size(0), device=x.device)

    delta.requires_grad_(True)
    preds = model(x + delta)
    loss = F.cross_entropy(preds, y)
    grad = torch.autograd.grad(loss, delta)[0].detach()
    
    if not warm_up:
        with torch.no_grad():
            grad_norm = torch.norm(grad.view(len(grad), -1), dim=1).detach() ** 2
            moving_grad_norm = beta * moving_grad_norm + (1 - beta) * grad_norm
            step_size = gamma_over_c / (1 + torch.sqrt(moving_grad_norm) / c)
            step_size = torch.clamp(step_size, min_step_size, max_step_size)
        # After computing per-sample scalar step_size (shape: [B])
        step_size = step_size.view(-1, 1, 1, 1)           # B × 1 × 1 × 1 (batch-specific scalar)
        step_size = step_size / std.view(1, -1, 1, 1)     # scale per channel (B × C × 1 × 1)
        step_size = step_size.expand_as(grad)             # B × C × H × W for broadcast in update

    else:
        step_size = eps
    delta = delta.detach() + step_size * torch.sign(grad.detach())
    delta = torch.clamp(delta, min=-eps, max=eps)
    delta = torch.clamp(delta, min=lower_limit - x, max=upper_limit - x)
    delta = delta.detach()

    if model_training:
        model.train()
    
    step_size = step_size.mean().item()
    return delta, grad, moving_grad_norm, step_size