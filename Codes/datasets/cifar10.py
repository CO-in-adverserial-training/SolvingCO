import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# CIFAR10 mean and std
norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2471, 0.2435, 0.2616)

# Transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


def get_loaders(batch_size: int = 128, num_workers: int = 2, device: str = 'cuda'):
    mu = torch.tensor(norm_mean).view(3,1,1).to(device)
    std = torch.tensor(norm_std).view(3,1,1).to(device)

    # Download the dataset
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    return trainloader, testloader, upper_limit, lower_limit, std, 10