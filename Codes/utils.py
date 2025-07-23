import torch
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