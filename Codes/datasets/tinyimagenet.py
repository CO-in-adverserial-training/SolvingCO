import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import urllib.request
import zipfile
from datasets.index_dataset import IndexDataset

# TinyImageNet Dataset
def get_loaders(batch_size: int = 128, num_workers: int = 2, device: str = 'cuda', normalize_dataset: bool = True, index_dataset: bool = False):
    dataset_path = './data'
    if normalize_dataset:
        # Using standard ImageNet mean and std since TinyImageNet is a subset
        tinyimagenet_mean = [0.485, 0.456, 0.406]
        tinyimagenet_std = [0.229, 0.224, 0.225]
    else:
        tinyimagenet_mean = [0., 0., 0.]
        tinyimagenet_std = [1., 1., 1.]

    mu = torch.tensor(tinyimagenet_mean).view(3,1,1).to(device)
    std = torch.tensor(tinyimagenet_std).view(3,1,1).to(device)

    upper_limit = (1 - mu) / std
    lower_limit = (0 - mu) / std
    
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tinyimagenet_mean, tinyimagenet_std),
        ])
    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(tinyimagenet_mean, tinyimagenet_std),
        ])
    
    
    # Create Datasets
    trainset = TinyImageNetDataset(root=dataset_path, train=True, transform=train_transform)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = TinyImageNetDataset(root=dataset_path, train=False, transform=test_transform)
    
    # Create Dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    
    # Get Class Names (simplified - actual class names would be from the wnids.txt file)
    classes = [f'class_{i}' for i in range(len(classes))]  # Replace with actual class names if needed
    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes), len(trainset), len(testset)


# Custom Dataset Class for Tiny ImageNet
class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.split = 'train' if train else 'val'
        
        # Paths
        self.data_dir = os.path.join(root, 'tiny-imagenet-200', self.split)
        if not os.path.exists(self.data_dir):
            self.download_and_prepare()
            
        # Load images and labels
        self.images = []
        self.labels = []
        
        if self.train:
            # For training data, images are in class folders
            class_dirs = [d for d in os.listdir(self.data_dir) 
                            if os.path.isdir(os.path.join(self.data_dir, d))]
            class_dirs.sort()
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}
            
            for class_name in class_dirs:
                class_dir = os.path.join(self.data_dir, class_name, 'images')
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        else:
            # For validation data, we need to read the val_annotations.txt file
            with open(os.path.join(self.data_dir, 'val_annotations.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    class_name = parts[1]
                    self.images.append(os.path.join(self.data_dir, 'images', img_name))
                    self.labels.append(class_name)
            
            # Convert validation labels to indices
            unique_classes = sorted(list(set(self.labels)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
            self.labels = [self.class_to_idx[label] for label in self.labels]
            
    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.images)
    
    # def download_and_prepare(self):
    #     # Download and extract Tiny ImageNet
    #     os.makedirs(self.root, exist_ok=True)
    #     !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P {self.root}
    #     !unzip -q {os.path.join(self.root, 'tiny-imagenet-200.zip')} -d {self.root}
    #     !rm {os.path.join(self.root, 'tiny-imagenet-200.zip')}

    def download_and_prepare(self):
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, 'tiny-imagenet-200.zip')
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

        if not os.path.exists(zip_path):
            print("Downloading Tiny ImageNet...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")

        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        print("Extraction complete.")

        os.remove(zip_path)
        print("Removed zip file.")