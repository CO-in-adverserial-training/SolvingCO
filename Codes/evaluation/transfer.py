import torch
import numpy as np
from .attacks import create_attack
from matplotlib import pyplot as plt

def plot_transfer(accuracies, clean, sdir, title=''):
    n_epochs = len(accuracies)

    # Create a figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Slightly wider figure

    # First plot (left)
    im1 = ax1.imshow(accuracies)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title(f'{title} Transferability')
    ax1.set_xticks(range(n_epochs), range(1, n_epochs+1))
    ax1.set_xlabel('Model')
    ax1.set_yticks(range(n_epochs), range(1, n_epochs+1))
    ax1.set_ylabel('Proxy')

    # Second plot (right)
    im2 = ax2.imshow(accuracies - clean, cmap='RdYlGn')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Accuracy Change')  # Optional: Add label to colorbar
    ax2.set_title(f'{title} Transferability (Accuracy Change)')
    ax2.set_xticks(range(n_epochs), range(1, n_epochs+1))
    ax2.set_xlabel('Model')
    ax2.set_yticks(range(n_epochs), range(1, n_epochs+1))
    ax2.set_ylabel('Proxy')

    plt.tight_layout()
    plt.savefig(f'{sdir}/transfer_{title}.pdf', bbox_inches='tight')
    plt.show()


def test_step(model, attack, loader, num_iters: int = -1, device: str = 'cuda'):
    model.eval()
    model.to(device)
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

def transfer_test_step(model1, model2, attack, loader, num_iters: int = -1, device: str = 'cuda'):
    model1.eval(), model2.eval()
    model1.to(device), model2.to(device)
    total, correct = 0, 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        img += attack(model1, img, lbl)
        pred = model2(img)
        total += lbl.shape[0]
        correct += sum(torch.argmax(pred, axis=1) == lbl).item()
        num_iters -= 1
        if num_iters == 0:
            break
    return 100 * correct / total

def test(architecture, saving_directory, loader, lower_limit, upper_limit, std, num_classes: int = 10, num_iters: int = -1, n_epochs: int = 30, device: str = 'cuda'):
    model1 = architecture(num_classes=num_classes).to(device)
    model2 = architecture(num_classes=num_classes).to(device)
    
    # Setup the attacks
    fgsm = create_attack(eps=8/255, alpha_coef=2, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=1, random_start=False, std=std)
    pgd = create_attack(eps=8/255, alpha_coef=0.25, lower_limit=lower_limit, upper_limit=upper_limit, attack_iters=10, random_start=True, std=std)

    metrics = {'Clean': np.zeros(n_epochs), 'FGSM': np.zeros((n_epochs, n_epochs)), 'PGD': np.zeros((n_epochs, n_epochs))}
    for i in range(n_epochs):
        # Load first model
        model1.load_state_dict(torch.load(f'{saving_directory}/weights_{str(i+1).zfill(2)}.pth', weights_only=False))

        # Measure clean accuracy and self trasnfer
        metrics['Clean'][i] = test_step(model1, None, loader, num_iters)
        metrics['FGSM'][i,i] = test_step(model1, fgsm, loader, num_iters)
        metrics['PGD'][i,i] = test_step(model1, pgd, loader, num_iters)

        for j in range(i):
            # Load second model
            model2.load_state_dict(torch.load(f'{saving_directory}/weights_{str(j+1).zfill(2)}.pth', weights_only=False))

            # Measure cross-transfer
            metrics['FGSM'][i,j] = transfer_test_step(model1, model2, fgsm, loader, num_iters)
            metrics['FGSM'][j,i] = transfer_test_step(model2, model1, fgsm, loader, num_iters)
            metrics['PGD'][i,j] = transfer_test_step(model1, model2, pgd, loader, num_iters)
            metrics['PGD'][j,i] = transfer_test_step(model2, model1, pgd, loader, num_iters)

        # Progress bar
        print(f'Epoch {str(i+1).zfill(2)}, {(i+1) ** 2} out of {n_epochs ** 2} done.')
    
    # Plots
    plot_transfer(metrics['FGSM'], metrics['Clean'], saving_directory, 'FGSM')
    plot_transfer(metrics['PGD'], metrics['Clean'], saving_directory, 'PGD')

    return metrics