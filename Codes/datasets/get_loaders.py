from .cifar10 import get_loaders as cifar10_loaders
from .cifar100 import get_loaders as cifar100_loaders
from .svhn import get_loaders as svhn_loaders
from .cinic10 import get_loaders as cinic10_loaders
from .tinyimagenet import get_loaders as tinyimagenet_loaders

def get_loaders(dataset_name: str, batch_size: int = 128, num_workers: int = 2,
                 device: str = 'cuda', normalize_dataset: bool = True, index_dataset: bool = False):
    match dataset_name:
        case "CIFAR10":
            return cifar10_loaders(batch_size, num_workers, device, normalize_dataset, index_dataset)
        case "CIFAR100":
            return cifar100_loaders(batch_size, num_workers, device, normalize_dataset, index_dataset)
        case "SVHN":
            return svhn_loaders(batch_size, num_workers, device, normalize_dataset, index_dataset)
        case "CINIC10":
            return cinic10_loaders(batch_size, num_workers, device, normalize_dataset, index_dataset)
        case "Tiny ImageNet":
            return tinyimagenet_loaders(batch_size, num_workers, device, normalize_dataset, index_dataset)
        case _:
            raise "Invalid Dataset!"