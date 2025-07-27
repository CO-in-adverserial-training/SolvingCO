import torch

def get_optimizer(args, model):
    match args.optimizer:
        case "SGD":
            return torch.optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        case _:
            raise "Invalid Optimizer!"

def get_scheduler(args, optimizer):
    match args.scheduler:
        case "Cyclic": # Default
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                                                  step_size_up=scheduler_up_iters, step_size_down=scheduler_down_iters)
        case "CosineAnnealing": # For TinyImageNet
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * LEN_TRAINLOADER, eta_min=0.001) 
        case "MultiStep": # For Runs With 110 Epochs
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        case _:
            raise "Invalid Scheduler!"