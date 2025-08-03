import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from timm import create_model


def VitBasePatch16_224(num_classes=10, img_size=32):
    """
    Get the model for the ViT-Base-Patch16-224 architecture.
    
    Args:
        num_classes (int): Number of classes in the dataset.
        img_size (int): Size of the image.
    """
    model = create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size 
            # img_size=32  # CIFAR-10 image size
            # img_size=224  # ImageNet image size
        )
    
    return model