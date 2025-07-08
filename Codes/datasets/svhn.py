import torch
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
import torchvision.transforms as transforms

# SVHN mean and std
norm_mean = (0.4380, 0.4440, 0.4730)
norm_std = (0.1751, 0.1771, 0.1744)

# Transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
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
    trainset = SVHN(root='./data', split='train', download=True, transform=transform_train)
    testset = SVHN(root='./data', split='test', download=True, transform=transform_train)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    return trainloader, testloader, upper_limit, lower_limit, std, 10