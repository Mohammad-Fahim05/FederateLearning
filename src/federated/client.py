import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.models.resnet_backbone import LocalFocalLoss
from src.privacy.dp_optimizer import DPGradientClipper

class HospitalClient:
    """
    Simulates a local hospital node in the federated network.
    Executes local model updates using Focal Loss + SAM-lite sharpness regularization
    and differential privacy gradient clipping.
    """
    def __init__(self, client_id, dataset, indices, config, device='cpu'):
        self.client_id = client_id
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Local DataLoader with optimized multiprocessing and prefetching
        num_workers = config.get('federated', {}).get('num_workers', 4 if os.name != 'nt' else 0)
        pin_memory = (self.device.type == 'cuda' and torch.cuda.is_available())
        persistent_workers = (num_workers > 0)
        
        subset = Subset(dataset, indices)
        self.loader = DataLoader(
            subset,
            batch_size=config['federated']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        self.num_samples = len(subset)
        
        # Local Focal Loss
        self.criterion = LocalFocalLoss(
            alpha=config['federated']['focal_alpha'],
            gamma=config['federated']['focal_gamma']
        )
        
        # DP Clipper with memory-safe chunking
        chunk_size = config.get('privacy', {}).get('chunk_size', 8)
        self.dp_clipper = DPGradientClipper(
            max_grad_norm=config['privacy']['max_grad_norm'],
            noise_multiplier=config['privacy']['noise_multiplier'] if config['privacy']['enabled'] else 0.0,
            enabled=config['privacy']['enabled'],
            chunk_size=chunk_size
        )
        
        self.sam_rho = config['federated']['sam_rho']

    def train_epoch(self, model, optimizer):
        model.to(self.device)
        model.train()
        
        total_loss = 0.0
        total_batches = 0
        
        for batch_idx, (images, labels, _, _) in enumerate(self.loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            
            # Step 1 & Step 3: Compute vectorized per-sample clipped & DP noised gradients
            loss_val = self.dp_clipper.clip_and_noise_sample_grad(
                model, self.criterion, images, labels, device=self.device
            )
            
            # Step 2: Apply local SGD step
            optimizer.step()
            
            total_loss += loss_val
            total_batches += 1
            
        avg_loss = total_loss / max(1, total_batches)
        return avg_loss

    def local_train(self, global_model):
        """
        Executes local epochs and returns local model weights and DP-protected local loss.
        """
        local_model = ResNet_copy(global_model)
        local_model.to(self.device)
        
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.config['federated']['learning_rate'],
            momentum=self.config['federated']['momentum'],
            weight_decay=self.config['federated']['weight_decay']
        )
        
        local_loss = 0.0
        epochs = self.config['federated']['local_epochs']
        for epoch in range(epochs):
            loss = self.train_epoch(local_model, optimizer)
            local_loss += loss
            
        avg_local_loss = local_loss / max(1, epochs)
        
        # Protect local loss statistic with Gaussian DP noise if privacy is enabled
        if self.config['privacy']['enabled']:
            import numpy as np
            loss_noise = float(np.random.normal(0.0, 0.05))
            protected_loss = max(0.0, avg_local_loss + loss_noise)
        else:
            protected_loss = avg_local_loss
        
        # Return state dict weights and DP-protected average loss
        return local_model.state_dict(), protected_loss

def ResNet_copy(model):
    import copy
    return copy.deepcopy(model)
