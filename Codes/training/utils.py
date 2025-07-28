import torch
from collections import defaultdict

# Class for tracking metrics for visualization and other processing
class MetricTracker:
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

    def get_history(self):
        return dict(self.data)
    
    def to_dict(self):
        return {key: self.history[key] for key in self.history}

# Get optimizer for model given name
def get_optimizer(args, model):
    match args.optimizer:
        case "SGD":
            return torch.optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        case _:
            raise ValueError("Invalid Optimizer!")
        
# Get scheduler for learning rate given name
def get_scheduler(args, optimizer, len_trainloader):
    match args.scheduler:
        case "Cyclic": # Default
            scheduler_up_iters = max((args.epochs * len_trainloader) // 2, 1)
            scheduler_down_iters = max(args.epochs * len_trainloader - (args.epochs * len_trainloader) // 2, 1)
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.2,
                                                  step_size_up=scheduler_up_iters, step_size_down=scheduler_down_iters)
        case "CosineAnnealing": # For TinyImageNet
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len_trainloader, eta_min=0.001) 
        case "MultiStep": # For Runs With 110 Epochs
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 105], gamma=0.1)
        case _:
            raise ValueError("Invalid Scheduler!")

# Returns dimensions of the data
def get_input_dimensions(dataloader, index_dataset):
    detailer = iter(dataloader)
    data = next(detailer)
    if index_dataset:
        images, _, _ = data
    else:
        images, _ = data

    return images.shape

def calculate_batch_accuracy(logits, labels):
    indices = torch.argmax(logits, 1)
    correct_count = (indices == labels).sum()
    return correct_count / labels.size(0)