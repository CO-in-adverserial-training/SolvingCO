import torch
from collections import defaultdict

class MetricTracker:
    """
    Class for tracking metrics for visualization and other processing.
    """
    def __init__(self):
        self.data = defaultdict(list)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.data[key].append(value)

    def average(self, key):
        return sum(self.data[key]) / len(self.data[key]) if self.data[key] else 0.0

    def result(self):
        return {k: self.average(k) for k in self.data}

    def reset(self):
        self.data = defaultdict(list)

    def to_dict(self):
        return dict(self.data)

def get_optimizer(args, model):
    """
    Get optimizer for model given name.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        model (torch.nn.Module): Model to optimize.
    """
    match args.optimizer:
        case "SGD":
            return torch.optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        case _:
            raise ValueError("Invalid Optimizer!")

def get_scheduler(args, optimizer, len_trainloader):
    """
    Get scheduler for learning rate given name.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        len_trainloader (int): Length of the training data loader.
    """
    match args.scheduler:
        case "Cyclic": # Default
            scheduler_up_iters = max((args.epochs * len_trainloader) // 2, 1)
            scheduler_down_iters = max(args.epochs * len_trainloader - (args.epochs * len_trainloader) // 2, 1)
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.initial_lr, max_lr=0.2,
                                                  step_size_up=scheduler_up_iters, step_size_down=scheduler_down_iters)
        case "CosineAnnealing": # For TinyImageNet
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len_trainloader, eta_min=0.001) 
        case "MultiStep": # For Runs With 110 Epochs
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 105], gamma=0.1)
        case _:
            raise ValueError("Invalid Scheduler!")

def get_input_dimensions(dataloader, index_dataset):
    """
    Get the dimensions of the input data.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Data loader to get the dimensions from.
        index_dataset (bool): Whether the dataset is indexed.
    """
    detailer = iter(dataloader)
    data = next(detailer)
    if index_dataset:
        images, _, _ = data
    else:
        images, _ = data

    return images.shape

def calculate_batch_accuracy(logits, labels):
    """
    Calculate the accuracy of the model on a batch of data.
    
    Args:
        logits (torch.Tensor): Logits of the model.
        labels (torch.Tensor): Labels of the data.
    """
    indices = torch.argmax(logits, 1)
    correct_count = (indices == labels).sum()
    return correct_count / labels.size(0)