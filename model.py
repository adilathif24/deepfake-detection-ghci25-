#!/usr/bin/env python3
"""
Deepfake Detection Model Architecture
Neural network model for detecting deepfakes using convolutional layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeepfakeDetectionModel(nn.Module):
    """CNN-based model for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize model architecture
        
        Args:
            num_classes: Number of output classes (real/fake)
            pretrained: Use pre-trained weights
        """
        super(DeepfakeDetectionModel, self).__init__()
        
        # Use ResNet50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom classification head
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Classification head
        logits = self.fc_layers(features)
        
        return logits
    
    def get_features(self, x):
        """Extract feature representation from backbone"""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class LightweightDeepfakeDetector(nn.Module):
    """Lightweight model for efficient deepfake detection on edge devices"""
    
    def __init__(self, num_classes=2):
        super(LightweightDeepfakeDetector, self).__init__()
        
        # Lightweight convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through lightweight model
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output logits
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test full model
    model = DeepfakeDetectionModel().to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    print(f"Full Model Output Shape: {output.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test lightweight model
    lightweight_model = LightweightDeepfakeDetector().to(device)
    output_light = lightweight_model(x)
    print(f"\nLightweight Model Output Shape: {output_light.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in lightweight_model.parameters())}")
