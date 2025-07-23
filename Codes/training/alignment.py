import torch.nn.functional as F

def calc_alignment(input_grad, adv_images):
    backprop_grad = adv_images.grad.clone().detach()
    return F.cosine_similarity(backprop_grad.view(backprop_grad.shape[0], -1), input_grad.view(input_grad.shape[0], -1)).mean().item()