import os
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CyclicLR


def train_epoch(model, optimizer, scheduler, loader, criterion, device: str = 'cuda'):
    batch_losses = []
    batch_accuracies = []
    total, correct = 0, 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        pred = model(img)
        optimizer.zero_grad()
        loss = criterion(pred, lbl)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Loss 
        batch_losses.append(loss.item())
        # Accuracy
        batch_correct = sum(torch.argmax(pred, axis=1) == lbl).item()
        batch_size = lbl.shape[0]
        correct += batch_correct
        total += batch_size
        batch_accuracies.append(100 * batch_correct / batch_size)
    epcoh_accuracy = 100 * correct / total
    return batch_losses, batch_accuracies, epcoh_accuracy


def train(architecture, loader, seed: int = 0, n_epochs: int = 30, device: str = 'cuda'):
    # Setting the seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = architecture().to(device)
    
    # Create saving directory
    saving_dir = f'experiments/{loader.dataset.__class__.__name__}_{model.__class__.__name__}_Benign_E{n_epochs}_S{seed}'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Setup optimizer and scheduler 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_steps = n_epochs * len(loader) / 2
    scheduler = CyclicLR(optimizer, base_lr=0., max_lr=0.2, step_size_up=lr_steps, step_size_down=lr_steps)  
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    losses, accuracies = [], []
    torch.save(model.state_dict(), f'{saving_dir}/weights_00.pth')

    for epoch in range(n_epochs):
        batch_losses, batch_accuracies, epcoh_accuracy = train_epoch(model, optimizer, scheduler, loader, criterion, device)
        print(f'Epcoh {str(epoch+1).zfill(2)}: Accuracy = {epcoh_accuracy:.2f}%')
        losses += batch_losses
        accuracies += batch_accuracies
        torch.save(model.state_dict(), f'{saving_dir}/weights_{str(epoch+1).zfill(2)}.pth')

    losses, accuracies = np.array(losses), np.array(accuracies)
    np.save(f'{saving_dir}/losses.npy', losses)
    np.save(f'{saving_dir}/accuracies.npy', accuracies)
    return saving_dir
