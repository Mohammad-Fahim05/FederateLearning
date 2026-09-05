import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18Backbone(nn.Module):
    """
    ResNet-18 classification model tailored for Camelyon17 (96x96 RGB patches).
    Supports feature extraction and domain-generalized binary classification.
    """
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Replace fully connected layer for binary classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Feature extractor
        self.classifier = nn.Linear(in_features, num_classes)
        
    def extract_features(self, x):
        return self.resnet(x)
        
    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

class LocalFocalLoss(nn.Module):
    """
    Focal Loss to combat severe local class imbalance across hospital patches.
    L = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Supports reduction='mean', 'sum', or 'none'.
    """
    def __init__(self, alpha=[0.5, 0.5], gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        alpha_t = self.alpha.to(logits.device)[targets]
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.mean()
