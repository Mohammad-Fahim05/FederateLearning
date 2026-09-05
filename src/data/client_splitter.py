import numpy as np
import torch
from torch.utils.data import Subset

class HospitalClientSplitter:
    """
    Splits Camelyon17 data into federated hospital clients.
    Implements natural hospital partitioning and synthetic heterogeneity scenarios (S1-S4).
    """
    def __init__(self, dataset, train_centers=[0, 1, 2], seed=42):
        self.dataset = dataset
        self.train_centers = train_centers
        self.seed = seed
        np.random.seed(seed)
        
    def get_natural_split(self):
        """
        Scenario S0: Natural hospital heterogeneity.
        1 Hospital Center = 1 Federated Client.
        """
        client_indices = {}
        for idx, center in enumerate(self.train_centers):
            indices = self.dataset.get_center_subsets([center])
            client_indices[idx] = indices
        return client_indices
        
    def get_label_shift_split(self, target_ratios={0: 0.1, 1: 0.5, 2: 0.9}):
        """
        Scenario S2: Label Shift / Class Imbalance.
        Client 0: 10% Tumor, Client 1: 50% Tumor, Client 2: 90% Tumor.
        """
        client_indices = {}
        for idx, center in enumerate(self.train_centers):
            indices = self.dataset.get_center_subsets([center])
            
            # Subsample according to desired tumor ratio
            pos_indices = []
            neg_indices = []
            for i in indices:
                _, y, _, _ = self.dataset[i]
                if y == 1:
                    pos_indices.append(i)
                else:
                    neg_indices.append(i)
                    
            desired_pos_ratio = target_ratios.get(idx, 0.5)
            # Adjust subset size
            min_size = min(len(pos_indices), len(neg_indices))
            if desired_pos_ratio > 0.5:
                n_pos = min_size
                n_neg = int(min_size * (1 - desired_pos_ratio) / desired_pos_ratio)
            else:
                n_neg = min_size
                n_pos = int(min_size * desired_pos_ratio / (1 - desired_pos_ratio))
                
            selected = pos_indices[:n_pos] + neg_indices[:n_neg]
            np.random.shuffle(selected)
            client_indices[idx] = selected
            
        return client_indices

    def get_size_imbalance_split(self, size_proportions=[0.7, 0.2, 0.1]):
        """
        Scenario S3: Client Size Imbalance.
        Client 0: 70% of patches, Client 1: 20%, Client 2: 10%.
        """
        client_indices = {}
        for idx, center in enumerate(self.train_centers):
            indices = self.dataset.get_center_subsets([center])
            prop = size_proportions[idx]
            sub_len = int(len(indices) * prop)
            np.random.shuffle(indices)
            client_indices[idx] = indices[:sub_len]
        return client_indices
