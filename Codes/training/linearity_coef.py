import torch
import torch.nn.functional as F

def calc_linearity_coef(input_grad, backprop_grad, method: str):
    match method:
        case "Second Order Theory":
            norm2_input_grad = torch.linalg.norm(input_grad.view(input_grad.shape[0], -1), dim=1, ord=2)
            linearity_coef = F.cosine_similarity(backprop_grad.view(backprop_grad.shape[0], -1), input_grad.view(input_grad.shape[0], -1)) / torch.pow(norm2_input_grad, 2)
            return linearity_coef.mean().item()
        case "Second Order Theory Sign":
            norm1_input_grad = torch.linalg.norm(input_grad.view(input_grad.shape[0], -1), dim=1, ord=1)
            linearity_coef = F.cosine_similarity(backprop_grad.view(backprop_grad.shape[0], -1), torch.sign(input_grad).view(input_grad.shape[0], -1)) / norm1_input_grad
            return linearity_coef.mean().item()
        case _:
            return None
