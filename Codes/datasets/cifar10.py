import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from datasets.index_dataset import IndexDataset

def get_loaders(batch_size: int, num_workers: int, normalize_dataset: bool, index_dataset: bool, root_path: str, device):
    if normalize_dataset:
        cifar10_mean = [0.4914, 0.4822, 0.4465] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        cifar10_std = [0.2471, 0.2435, 0.2616] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        cifar10_mean = [0., 0., 0.]
        cifar10_std = [1., 1., 1.]
    
    mu = torch.tensor(cifar10_mean).view(3,1,1).to(device)
    std = torch.tensor(cifar10_std).view(3,1,1).to(device)
    
        
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    

    # Download the dataset
    trainset = CIFAR10(root=f'{root_path}/data', train=True, download=True, transform=train_transform)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = CIFAR10(root=f'{root_path}/data', train=False, download=True, transform=test_transform)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    # Name of Classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes), len(trainset), len(testset)