import torch
from matplotlib import pyplot as plt
from datasets.get_loaders import get_loaders
from attacks.fgsm import fgsm
from attacks.pgd import pgd
from utils import load_checkpoint


def test_step(model, attack, loader, upper_limit, lower_limit, mu, std, epsilon, device):
    model.eval()
    total, correct = 0, 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        if attack:
            if attack == "FGSM":
                delta, _ = fgsm(model, img, lbl, upper_limit, lower_limit, mu, std, epsilon, 2 * epsilon)
            elif attack == "PGD":
                delta, _ = pgd(model, img, lbl, upper_limit, lower_limit, mu, std, epsilon, epsilon / 4)
            else:
                raise ValueError("Invalid Attack!")
            img += delta
            model.eval()
        pred = model(img)
        total += lbl.shape[0]
        correct += sum(torch.argmax(pred, axis=1) == lbl).item()

    return 100 * correct / total

def test(args, device, max_eps: int = 32):
    # Get dataset loaders
    trainloader, testloader, upper_limit, lower_limit, mu, std, _, num_classes, _, num_test_samples = get_loaders(args, False, device)
    
    final_checkpoint_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/checkpoints/model{str(args.epochs).zfill(3)}.pt"
    
    model, _, _ = load_checkpoint(args, final_checkpoint_path, num_classes, len(trainloader), device)
    
    model.eval()
    
    clean_acc = test_step(model, None, testloader)
    fgsm_accs, pgd_accs = [], []

    for eps in range(1, max_eps + 1):
        # Attacks
        fgsm = create_attack(eps=eps/255, alpha_coef=2, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=1, random_start=False, std=std)
        pgd = create_attack(eps=eps/255, alpha_coef=0.1, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=50, random_start=True, std=std)
        # Perform the tests
        f_acc = test_step(model, fgsm, testloader)
        p_acc = test_step(model, pgd, testloader)
        fgsm_accs.append(f_acc), pgd_accs.append(p_acc)
        # Progress bar
        print(f'Epsilon {str(eps).zfill(2)}: Clean {clean_acc:.2f}% | FGSM {f_acc:.2f}% | PGD {p_acc:.2f}%')
    f_auc = (clean_acc + sum(fgsm_accs)) / (max_eps + 1)
    p_auc = (clean_acc + sum(pgd_accs)) / (max_eps + 1)

    return f_auc, p_auc
