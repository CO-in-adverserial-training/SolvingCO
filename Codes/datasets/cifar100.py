import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from datasets.index_dataset import IndexDataset

def get_loaders(batch_size: int = 128, num_workers: int = 2, device: str = 'cuda', normalize_dataset: bool = True, index_dataset: bool = False):
    if normalize_dataset:
        cifar100_mean = [0.5071, 0.4865, 0.4409] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        cifar100_std =  [0.2673, 0.2564, 0.2762] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        cifar100_mean = [0., 0., 0.]
        cifar100_std = [1., 1., 1.]
    
    mu = torch.tensor(cifar100_mean).view(3,1,1).to(device)
    std = torch.tensor(cifar100_std).view(3,1,1).to(device)
    
        
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    

    # Download the dataset
    trainset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    # Name of Classes
    classes = (
        'apple','aquarium_fish','baby',
        'bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle',
        'caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin',
        'elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard',
        'lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter',
        'palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum',
        'rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
        'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout',
        'tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm')

    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes), len(trainset), len(testset)