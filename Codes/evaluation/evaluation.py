from ..utils import load_checkpoint
from attacks.get_attack import get_attack
import torch.nn.functional as F

def evaluate(dataloader, checkpoints_path: str, model_name: str, num_classes: int, attack_name: str, attack_params, num_epochs: int=30, device:str="cuda"):
    reg_list_test = []
    
    use_regularizer = attack_name in ["TRADES", "GradAlign", "ELLE"]
    for epoch in range(num_epochs + 1):
        model, _, _ = load_checkpoint(model_name, num_classes, f"{checkpoints_path}/model{str(epoch).zfill(3)}.pt")
        model.eval()

        for i, data in enumerate(dataloader):
            if index_dataset:
                images, labels, index = data[0].to(device), data[1].to(device), data[2]
            else:
                images, labels = data[0].to(device), data[1].to(device)
            # Determine attack
            attack = get_attack(attack_name)
            match attack_name:
                case attack if attack in  ["FGSM", "FGSM-RS", "NFGSM", "ZeroGrad", "SIA", "PGD"]:
                    delta, grad = attack(model, images, labels, **attack_params)
                case attack if attack in ["TRADES", "GradAlign", "ELLE"]:
                    delta, reg, grad = attack(model, images, labels, **attack_params)
                case "ATAS":
                    delta, grad = attack(model, images, labels, index, **attack_params)
                    delta_atas[index] = delta.clone().detach()
                case "FGSM-EP":
                    delta, reg, grad = attack(model, images, labels, index, **attack_params)
                    delta_fgsm_ep[index] = delta.clone().detach()
                case _:
                    raise "Invalid Attack Method!"
            # Add perturbation to original images
            adv_images = images + delta
            # Forward pass with adversarial examples
            preds = model(adv_images)
            loss = F.cross_entropy(preds, labels)
            #Track Regularizer Value Per Batch
            if use_regularizer:
                reg = reg.cpu() if reg is not None else None
                reg = reg.item()
                reg_list_test.append(reg)