import torch
from pathlib import Path
from architectures.get_model import get_model
from training.utils import get_optimizer, get_scheduler

# Seed setting for result reproducibility
def set_seeds():
    pass # TODO

# Create necessary directories
def create_directories(root_path: str):
    Path(f'{root_path}/loss_plots').mkdir(parents=True, exist_ok=True)
    Path(f'{root_path}/acc_vs_eps_plots').mkdir(parents=True, exist_ok=True)
    Path(f'{root_path}/decision_boundry_plots').mkdir(parents=True, exist_ok=True)
    Path(f'{root_path}/tsne_plots').mkdir(parents=True, exist_ok=True)
    Path(f'{root_path}/layerwise_lipschitz_plots').mkdir(parents=True, exist_ok=True)
    Path(f'{root_path}/checkpoints').mkdir(parents=True, exist_ok=True)
    Path(f'{root_path}/data').mkdir(parents=True, exist_ok=True)

# Get device
def get_device(device_name):
    match device_name:
        case "cuda":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        case "cpu":
            return torch.device('cpu')
        case _:
            raise ValueError("Invalid Device!")

# Save model checkpoint
def save_checkpoint(model, optimizer, scheduler, path:str):
    torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                   }, path)

# Load model checkpoint
def load_checkpoint(args, num_classes:int, path:str, device):
    model = get_model(args.model, num_classes)
    model.to(device)
    
    optimizer = get_optimizer(args.optimizer, model)
    scheduler = get_scheduler(args.scheduler, optimizer, len_trainloader)

    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler