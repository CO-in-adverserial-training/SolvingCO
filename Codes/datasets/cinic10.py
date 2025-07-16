import torch
import torchvision
import torchvision.transforms as transforms
import tarfile
import requests
import os
from pathlib import Path
from .index_dataset import IndexDataset

def download_and_extract_cinic10(dest_path):
    url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    archive_path = os.path.join(dest_path, "CINIC-10.tar.gz")
    os.makedirs(dest_path, exist_ok=True)

    print("Downloading CINIC-10 (SSL verify disabled)...")
    response = requests.get(url, stream=True, verify=False)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: status code {response.status_code}")
    with open(archive_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_path)
    os.remove(archive_path)
    print("Download and extraction completed.")

# CINIC10 Dataset
def get_loaders(batch_size: int = 128, num_workers: int = 2, device: str = 'cuda', normalize_dataset: bool = True, index_dataset: bool = False):
    
    # CINIC-10 channel stats
    if normalize_dataset:
        cinic10_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic10_std = [0.24205776, 0.23828046, 0.25874835]
    else:
        cinic10_mean = [0., 0., 0.]
        cinic10_std = [1., 1., 1.]
    
    # Clamp Tensors
    mu = torch.tensor(cinic10_mean).view(3,1,1).to(device)
    std = torch.tensor(cinic10_std).view(3,1,1).to(device)
    upper_limit = (1 - mu) / std
    lower_limit = (0 - mu) / std
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cinic10_mean, cinic10_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(cinic10_mean, cinic10_std),
    ])
    
    local_path = "./data"
    if os.path.exists(os.path.join(local_path, 'train')) and os.path.exists(os.path.join(local_path, 'test')):
        print(f"Found local CINIC-10 dataset at: {local_path}")
        trainset = torchvision.datasets.ImageFolder(os.path.join(local_path, 'train'), transform=train_transform)
        testset = torchvision.datasets.ImageFolder(os.path.join(local_path, 'test'), transform=test_transform)
    else:
        # Fallback to download (though this won't work on Kaggle without internet)
        Path(local_path).mkdir(parents=True, exist_ok=True)
        download_and_extract_cinic10(local_path)

        trainset = torchvision.datasets.ImageFolder(f'{local_path}/train', transform=train_transform)
        testset = torchvision.datasets.ImageFolder(f'{local_path}/test', transform=test_transform)

    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, testset, trainloader, testloader, mu, std, upper_limit, lower_limit, classes, len(classes)
