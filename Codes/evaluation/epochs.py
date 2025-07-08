import torch
from .attacks import create_attack
from matplotlib import pyplot as plt

def plot_test_acc(accuracies, n_epochs=30):
    xs = list(range(1, 1 + n_epochs))
    width = 4 * n_epochs // 10
    plt.figure(figsize=(width, 8))
    for key in accuracies.keys():
        plt.plot(xs, accuracies[key], label=key)
    plt.legend()
    plt.xticks(xs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
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

def test(architecture, saving_directory, loader, lower_limit, upper_limit, std, num_classes: int = 10, num_iters: int = -1, n_epochs: int = 30, device: str = 'cuda'):
    model = architecture(num_classes=num_classes).to(device)
    
    # Setup the attacks
    fgsm2 = create_attack(eps=2/255, alpha_coef=2, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=1, random_start=False, std=std)
    fgsm4 = create_attack(eps=4/255, alpha_coef=2, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=1, random_start=False, std=std)
    fgsm6 = create_attack(eps=6/255, alpha_coef=2, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=1, random_start=False, std=std)
    fgsm8 = create_attack(eps=8/255, alpha_coef=2, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=1, random_start=False, std=std)
    pgd = create_attack(eps=8/255, alpha_coef=0.25, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=10, random_start=True, std=std)

    metrics = {'Clean': [], 'FGSM with ε = 2': [], 'FGSM with ε = 4': [], 'FGSM with ε = 6': [], 'FGSM with ε = 8': [], 'PGD with ε = 8': []}
    for i in range(n_epochs):
        # Load model parameters for the current epoch
        model.load_state_dict(torch.load(f'{saving_directory}/weights_{str(i+1).zfill(2)}.pth', weights_only=False))
        model.eval()
        model.to(device)
        # Perform the tests
        c_acc = test_step(model, None, loader, num_iters)
        f_acc = test_step(model, fgsm8, loader, num_iters)
        p_acc = test_step(model, pgd, loader, num_iters)
        metrics['Clean'].append(c_acc), metrics['FGSM with ε = 8'].append(f_acc), metrics['PGD with ε = 8'].append(p_acc)
        metrics['FGSM with ε = 2'].append(test_step(model, fgsm2, loader, num_iters))
        metrics['FGSM with ε = 4'].append(test_step(model, fgsm4, loader, num_iters))
        metrics['FGSM with ε = 6'].append(test_step(model, fgsm6, loader, num_iters))
        # Progress bar
        print(f'Epoch {str(i+1).zfill(2)}: Clean {c_acc:.2f}% | FGSM {f_acc:.2f}% | PGD {p_acc:.2f}%')
    plot_test_acc(metrics, n_epochs)
    return metrics