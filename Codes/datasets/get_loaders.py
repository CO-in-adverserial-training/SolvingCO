from .cifar10 import get_loaders as cifar10_loaders
from .cifar100 import get_loaders as cifar100_loaders
from .svhn import get_loaders as svhn_loaders
from .cinic10 import get_loaders as cinic10_loaders
from .tinyimagenet import get_loaders as tinyimagenet_loaders
from .pathmnist import get_loaders as pathmnist_loaders

# Returns trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes)
def get_loaders(args, index_dataset: bool, device):
    match args.dataset:
        case "CIFAR10":
            return cifar10_loaders(args.batch_size, args.num_workers, device, args.normalize_dataset, index_dataset)
        case "CIFAR100":
            return cifar100_loaders(args.batch_size, args.num_workers, device, args.normalize_dataset, index_dataset)
        case "SVHN":
            return svhn_loaders(args.batch_size, args.num_workers, device, args.normalize_dataset, index_dataset)
        case "CINIC10":
            return cinic10_loaders(args.batch_size, args.num_workers, device, args.normalize_dataset, index_dataset)
        case "Tiny ImageNet":
            return tinyimagenet_loaders(args.batch_size, args.num_workers, device, args.normalize_dataset, index_dataset)
        case "PathMNIST":
            return pathmnist_loaders(args.batch_size, args.num_workers, device, args.normalize_dataset, index_dataset)
        case _:
            raise "Invalid Dataset!"