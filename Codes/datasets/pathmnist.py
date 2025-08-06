import torch
from torch.utils.data import DataLoader
from medmnist import PathMNIST
import torchvision.transforms as transforms
from datasets.index_dataset import IndexDataset

# Custom Dataset Wrapper to Squeeze Labels
class PathMNISTWrapper(torch.utils.data.Dataset):
    """
    A class to wrap the PathMNIST dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): The base dataset to wrap.
    """
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
def get_loaders(args, index_dataset: bool, device):
    """
    Get the loaders for the PathMNIST dataset.
    
    Args:
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of workers for the data loader.
        normalize_dataset (bool): Whether to normalize the dataset.
        index_dataset (bool): Whether to index the dataset.
        root_path (str): Path to the root directory of the project.
        device (torch.device): Device to use for the training.
    """
    
    if args.normalize_dataset:
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
    trainset = PathMNIST(root=f'{args.root_path}/Datasets/{args.dataset}', split='train', transform=train_transform, download=True, size=28)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = PathMNIST(root=f'{args.root_path}/Datasets/{args.dataset}', split='test', transform=test_transform, download=True, size=28)
    
    # Wrap datasets to fix label shape
    trainset = PathMNISTWrapper(trainset)
    testset = PathMNISTWrapper(testset)
    
    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)

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