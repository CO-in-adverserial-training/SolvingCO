import torch
from torch.utils.data import DataLoader
from medmnist import PathMNIST, TissueMNIST, OrganAMNIST, BloodMNIST
import torchvision.transforms as transforms
from datasets.index_dataset import IndexDataset

# Custom Dataset Wrapper to Squeeze Labels
class MedMNISTWrapper(torch.utils.data.Dataset):
    """
    A class to wrap the MedMNIST dataset.
    
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
    Get the loaders for the MedMNIST dataset.
    
    Args:
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of workers for the data loader.
        normalize_dataset (bool): Whether to normalize the dataset.
        index_dataset (bool): Whether to index the dataset.
        root_path (str): Path to the root directory of the project.
        device (torch.device): Device to use for the training.
    """
    # Map dataset name to class and metadata
    dataset_map = {
        'PathMNIST': {
            'class': PathMNIST,
            'channels': 3,
            'num_classes': 9,
            'mean': [0., 0., 0.],
            'std': [1., 1., 1.],
            'classes': [
                'adipose', 'background', 'debris', 'lymphocytes', 'mucus',
                'smooth muscle', 'normal colon', 'cancer-associated stroma', 'colorectal adenocarcinoma'
            ]
        },
        'TissueMNIST': {
            'class': TissueMNIST,
            'channels': 1,
            'num_classes': 8,
            'mean': [0.],
            'std': [1.],
            'classes': [
                'cortex', 'glomeruli', 'medulla', 'blood vessels',
                'pelvis', 'calyces', 'fat', 'background'
            ]
        },
        'OrganAMNIST': {
            'class': OrganAMNIST,
            'channels': 1,        # single‐channel CT slices
            'num_classes': 11,
            'mean': [0.],
            'std': [1.],
            'classes': [
                'spleen',
                'right kidney',
                'left kidney',
                'gallbladder',
                'esophagus',
                'liver',
                'stomach',
                'aorta',
                'pancreas',
                'right adrenal gland',
                'left adrenal gland'
            ]
        },
        'BloodMNIST': {                
            'class': BloodMNIST,
            'channels': 3,              # blood-cell microscope images are RGB
            'num_classes': 8,
            'mean': [0., 0., 0.],
            'std': [1., 1., 1.],
            'classes': [               # exact names from INFO in medmnist/info.py
                'erythrocyte',
                'eosinophil granulocyte',
                'large unstained cell',
                'lymphocyte',
                'monocyte',
                'neutrophil granulocyte',
                'basophil granulocyte',
                'platelet'
            ]
        }
    }

    # Validate input
    if args.dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    ds_info = dataset_map[args.dataset]
    to_rgb = (ds_info['channels'] == 1)

    # Define transforms
    def build_transform(train=True):
        ops = [transforms.ToTensor()]
        if to_rgb:
            # replicate single channel → 3 channels
            ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        if train:
            ops += [
                transforms.Pad(2),
                transforms.RandomRotation(10),
                transforms.Normalize(mean=ds_info['mean'], std=ds_info['std'])
            ]
        else:
            ops += [
                transforms.Pad(2),
                transforms.Normalize(mean=ds_info['mean'], std=ds_info['std'])
            ]
        return transforms.Compose(ops)


    if args.normalize_dataset:
        medmnist_mean = dataset_map[args.dataset]['mean'] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        medmnist_std = dataset_map[args.dataset]['std'] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        if args.dataset in ["TissueMNIST", "OrganAMNIST"]:
            medmnist_mean = [0.]
            medmnist_std = [1.]
        else:
            medmnist_mean = [0., 0., 0.]
            medmnist_std = [1., 1., 1.]

    mu = torch.tensor(medmnist_mean).view(3,1,1).to(device)
    std = torch.tensor(medmnist_std).view(3,1,1).to(device)
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # Flip for chest X-rays
        transforms.Pad(2),
        transforms.RandomRotation(degrees=10),   # Small rotations
        transforms.Normalize(medmnist_mean, medmnist_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize(medmnist_mean, medmnist_std)
    ])

    train_transform = build_transform(train=True)
    test_transform  = build_transform(train=False)

    # Load the actual datasets
    trainset = ds_info['class'](
        root=f'{args.root_med}/Datasets/{args.dataset}',
        split='train', transform=train_transform,
        download=True, size=28
    )
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset  = ds_info['class'](
        root=f'{args.root_med}/Datasets/{args.dataset}',
        split='test',  transform=test_transform,
        download=True, size=28
    )

    # Wrap datasets to fix label shape
    trainset = MedMNISTWrapper(trainset)
    testset = MedMNISTWrapper(testset)
    
    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)
    
    return trainloader, testloader, upper_limit, lower_limit, mu, std, ds_info['classes'], ds_info['num_classes'], len(trainset), len(testset)