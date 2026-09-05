import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

def get_transforms(is_train=True, feature_shift=False):
    """
    Returns PyTorch torchvision transformations for Camelyon17 patches (96x96).
    Camelyon17 images are RGB tissue patches.
    """
    transform_list = []
    
    if is_train:
        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90)
        ])
        
        if feature_shift:
            # Synthetic stain variation / color jitter for Scenario S1 stress test
            transform_list.append(
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            )
            
    transform_list.extend([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return T.Compose(transform_list)

class StainMatrixAugmentation:
    """
    Simulates H&E stain variations across different hospital scanners.
    """
    def __init__(self, alpha_range=(0.8, 1.2), beta_range=(-0.1, 0.1)):
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def __call__(self, img_tensor):
        alpha = torch.empty(3, 1, 1).uniform_(*self.alpha_range)
        beta = torch.empty(3, 1, 1).uniform_(*self.beta_range)
        augmented = img_tensor * alpha + beta
        return torch.clamp(augmented, 0.0, 1.0)
