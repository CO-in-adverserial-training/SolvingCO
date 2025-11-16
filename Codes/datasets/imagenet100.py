import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets.index_dataset import IndexDataset  # your index wrapper

def get_loaders(args, index_dataset: bool, device):
    """
    Prepare ImageNet-100 (subset of ImageNet-1K) loaders.
    Assumes you have manually downloaded ImageNet-1K and generated the 100-class subset.
    """

    # Standard ImageNet normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]  # these are standard ImageNet values :contentReference[oaicite:3]{index=3}

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # Paths for your subset
    train_dir = os.path.join(args.root_path, "imagenet100", "train")
    val_dir   = os.path.join(args.root_path, "imagenet100", "val")

    # Make sure directories exist
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise RuntimeError(f"ImageNet-100 directory not found. Make sure you've generated it: {train_dir}, {val_dir}")

    # Create datasets
    train_set = datasets.ImageFolder(root=train_dir, transform=train_transform)
    if index_dataset:
        train_set = IndexDataset(train_set)
    val_set = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # DataLoaders
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    valloader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Normalization boundary tensors (for adversarial training)
    mu = torch.tensor(imagenet_mean).view(3, 1, 1).to(device)
    std = torch.tensor(imagenet_std).view(3, 1, 1).to(device)
    upper_limit = ((1.0 - mu) / std).to(device)
    lower_limit = ((0.0 - mu) / std).to(device)

    # Class names and counts
    classes = tuple(train_set.classes)
    num_classes = len(classes)
    len_train = len(train_set)
    len_val = len(val_set)

    return trainloader, valloader, upper_limit, lower_limit, mu, std, classes, num_classes, len_train, len_val
