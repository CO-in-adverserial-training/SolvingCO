import torch
from autoattack import AutoAttack
from matplotlib import pyplot as plt

def plot_attack_acc(accuracies, n_epochs=30):
    xs = list(range(1, 1 + n_epochs))
    width = 4 * n_epochs // 10
    plt.figure(figsize=(width, 8))
    for attack_name in accuracies.keys():
        plt.plot(xs, accuracies[attack_name], label=attack_name)
    plt.legend()
    plt.xticks(xs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Robustness Evaluation per Attack Type')
    plt.grid()
    plt.show()

def test_step(model, adversary, loader, num_iters: int = -1, device: str = 'cuda'):
    model.eval()
    total, correct = 0, 0
    
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        
        if adversary:
            # AutoAttack-specific evaluation
            x_adv = adversary.run_standard_evaluation(img, lbl, bs=img.size(0))
            pred = model(x_adv)  # Evaluate on adversarial examples
        else:
            pred = model(img)
            
        total += lbl.shape[0]
        correct += (torch.argmax(pred, axis=1) == lbl).sum().item()
        
        num_iters -= 1
        if num_iters == 0:
            break
            
    return 100 * correct / total

def evaluate_autoattack_components(model, loader, eps=8/255, num_iters=-1, device='cuda'):
    model.eval()
    attacks = {
        'APGD-CE': {'version': 'apgd-ce'},
        'APGD-DLR': {'version': 'apgd-dlr'},
        'FAB': {'version': 'fab'},
        'Square Attack': {'version': 'square'}
    }
    
    metrics = {'Clean': []}
    for attack_name in attacks.keys():
        metrics[attack_name] = []
    
    # Initialize AutoAttack once (we'll override its version later)
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=eps,
        version='standard',  # Placeholder
        device=device,
        verbose=False
    )
    
    # if 'fab' in adversary.attacks_to_run:
    #     adversary.fab.n_restarts = 5                # Increase from default (more attempts)
    #     adversary.fab.n_target_classes = 9          # For CIFAR-10 (num_classes-1)
    #     adversary.fab.max_iter = 100                # Default 100 (keep or increase)
    #     adversary.fab.verbose = True                # See real-time progress

    # Evaluate clean accuracy first
    clean_acc = test_step(model, None, loader, num_iters, device)
    metrics['Clean'].append(clean_acc)
    
    # Evaluate each attack component separately
    for attack_name, config in attacks.items():
        # Configure AutoAttack to run only the current attack
        adversary.attacks_to_run = [config['version']]
        
        # Run evaluation
        adv_acc = test_step(model, adversary, loader, num_iters, device)
        metrics[attack_name].append(adv_acc)
    
    # Evaluate the full AutoAttack (all components)
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square']
    full_autoattack_acc = test_step(model, adversary, loader, num_iters, device)
    metrics['AutoAttack'] = [full_autoattack_acc]
    
    return metrics

def test(architecture, saving_directory, loader, lower_limit, upper_limit, std, num_classes=10, n_epochs=30, eps=8/255, num_iters=-1, device='cuda'):
    model = architecture(num_classes=num_classes).to(device)
    metrics = {
        'Clean': [],
        'APGD-CE': [],
        'APGD-DLR': [],
        'FAB': [],
        'Square Attack': [],
        'AutoAttack': []
    }
    
    for epoch in range(1, n_epochs + 1):
        model.load_state_dict(torch.load(f'{saving_directory}/weights_{str(epoch).zfill(2)}.pth', weights_only=False))
        epoch_metrics = evaluate_autoattack_components(model, loader, eps=eps/std, num_iters=num_iters, device=device)
        
        # Append results for each attack
        for key in metrics.keys():
            metrics[key].append(epoch_metrics[key][0])
        
        # Print progress
        print(f'Epoch {epoch:02d}: ' + ' | '.join(
            [f'{k} {v[-1]:.2f}%' for k, v in metrics.items()]
        ))
    
    plot_attack_acc(metrics, n_epochs)
    return metrics
