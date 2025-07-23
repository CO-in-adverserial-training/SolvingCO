import torch
from pathlib import Path
from .architectures import get_model

# Save model checkpoint
def save_checkpoint(model, optimizer, scheduler, path:str):
    torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                   }, path)

# Load model checkpoint
def load_checkpoint(model_name, num_classes: int=10, path: str=None, device: str = 'cuda'):
    model = get_model(model_name, num_classes)
        
    model.to(device)
    
    optimizer = ... # TODO Implement a good method for retrieving optimizer
    scheduler = ... # TODO Implement a good method for retrieving scheduler

    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler

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
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')