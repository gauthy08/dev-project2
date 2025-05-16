import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, n_mels=128, output_size=50):
        super().__init__()
        # Input shape: [batch, 1, n_mels, time_steps]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Adaptive pooling to handle variable length inputs
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Ensure input is in the right shape [batch, channels, freq, time]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Apply convolution blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling and flatten
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x