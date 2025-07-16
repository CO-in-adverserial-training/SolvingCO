import torch
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
import torchvision.transforms as transforms

def get_loaders(batch_size: int = 128, num_workers: int = 2, device: str = 'cuda', normalize_dataset: bool = True):
    if normalize_dataset:
        # SVHN mean and std
        svhn_mean = [0.4380, 0.4440, 0.4730] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        svhn_std = [0.1751, 0.1771, 0.1744] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        svhn_mean = [0., 0., 0.]
        svhn_std = [1., 1., 1.]
    
    mu = torch.tensor(svhn_mean).view(3,1,1).to(device)
    std = torch.tensor(svhn_std).view(3,1,1).to(device)
    
    # Transformations
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(svhn_mean, svhn_std),
    ])
    

    # Download the dataset
    trainset = SVHN(root='./data', split='train', download=True, transform=train_transform)
    testset = SVHN(root='./data', split='test', download=True, transform=test_transform)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    # Name of Classes
    classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes)


