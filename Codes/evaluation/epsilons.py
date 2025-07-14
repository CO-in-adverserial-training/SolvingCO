import torch
from .attacks import create_attack
from matplotlib import pyplot as plt

def plot_accuracies(clean, fgsm, pgd, saving_dir):
    epsilons = list(range(1, 1 + len(fgsm)))  # [1, 2, 3,...]
    
    plt.figure(figsize=(24, 12))
    
    # Plot clean accuracy as horizontal line
    plt.axhline(y=clean, color='k', linestyle='--', linewidth=2, label='Clean')
    
    # Plot adversarial accuracies with markers
    plt.plot(epsilons, fgsm, 'o-', label='FGSM')
    plt.plot(epsilons, pgd, 's-', label='PGD')
    
    # Formatting
    plt.ylim(0-5, 100+5)  # Force 0-100% range
    plt.xticks(epsilons)
    plt.xlabel(r'Attack Strength ($\epsilon \times 255$)')
    plt.ylabel('Accuracy (%)')
    # plt.title('Model Robustness Across Attack Strengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{saving_dir}_acc_eps.pdf', bbox_inches='tight')
    plt.show()

def test_step(model, attack, loader, num_iters: int = -1, device: str = 'cuda'):
    model.eval()
    total, correct = 0, 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        if attack:
            img += attack(model, img, lbl)
            model.eval()
        pred = model(img)
        total += lbl.shape[0]
        correct += sum(torch.argmax(pred, axis=1) == lbl).item()
        num_iters -= 1
        if num_iters == 0:
            break
    return 100 * correct / total

def test(architecture, saving_directory, loader, lower_limit, upper_limit, std, num_classes: int = 10, num_iters: int = -1, max_eps: int = 32, epoch: int = 30, device: str = 'cuda'):
    model = architecture(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(f'{saving_directory}/weights_{str(epoch).zfill(2)}.pth', weights_only=False))
    model.eval()
    
    c_acc = test_step(model, None, loader, num_iters)
    f_accs, p_accs = [], []

    for eps in range(1, max_eps + 1):
        # Attacks
        fgsm = create_attack(eps=eps/255, alpha_coef=2, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=1, random_start=False, std=std)
        pgd = create_attack(eps=eps/255, alpha_coef=0.1, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=50, random_start=True, std=std)
        # Perform the tests
        f_acc = test_step(model, fgsm, loader, num_iters)
        p_acc = test_step(model, pgd, loader, num_iters)
        f_accs.append(f_acc), p_accs.append(p_acc)
        # Progress bar
        print(f'Epsilon {str(eps).zfill(2)}: Clean {c_acc:.2f}% | FGSM {f_acc:.2f}% | PGD {p_acc:.2f}%')
    f_auc = (c_acc + sum(f_accs)) / (max_eps + 1)
    p_auc = (c_acc + sum(p_accs)) / (max_eps + 1)
    plot_accuracies(c_acc, f_accs, p_accs, saving_directory)
    print(f'FGSM-AUC = {f_auc:.2f} | PGD-AUC {p_auc:.2f}')
    return f_auc, p_auc
