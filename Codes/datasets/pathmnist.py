import torch
from torch.utils.data import DataLoader
from medmnist import PathMNIST
import torchvision.transforms as transforms
from .index_dataset import IndexDataset

# Custom Dataset Wrapper to Squeeze Labels
class PathMNISTWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Squeeze the label to remove the extra dimension
        label = label.squeeze()  # Converts [1] to scalar
        return img, label

# MedMNIST Dataset
def get_loaders(batch_size: int = 128, num_workers: int = 2, device: str = 'cuda', normalize_dataset: bool = True, index_dataset: bool = False):
    if normalize_dataset:
        pathmnist_mean = ... # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        pathmnist_std = ... # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        pathmnist_mean = [0., 0., 0.]
        pathmnist_std = [1., 1., 1.]

    mu = torch.tensor(pathmnist_mean).view(3,1,1).to(device)
    std = torch.tensor(pathmnist_std).view(3,1,1).to(device)
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # Flip for chest X-rays
        transforms.Pad(2),
        transforms.RandomRotation(degrees=10),   # Small rotations
        transforms.Normalize(pathmnist_mean, pathmnist_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize(pathmnist_mean, pathmnist_std)
    ])
    
    # Load PathMNIST datasets
    trainset = PathMNIST(split='train', transform=train_transform, download=True, size=28)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = PathMNIST(split='test', transform=test_transform, download=True, size=28)
    
    # Wrap datasets to fix label shape
    trainset = PathMNISTWrapper(trainset)
    testset = PathMNISTWrapper(testset)
    
    # Create DataLoaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    classes = (
        'adipose', 
        'background', 
        'debris', 
        'lymphocytes', 
        'mucus', 
        'smooth muscle', 
        'normal colon', 
        'cancer-associated stroma', 
        'colorectal adenocarcinoma'
    )
    
    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes), len(trainset), len(testset)