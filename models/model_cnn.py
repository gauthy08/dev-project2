import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with projection if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out

class AudioCNN(nn.Module):
    def __init__(self, n_mels=128, output_size=50):
        super().__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(16, 32, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.res_block2 = ResidualBlock(32, 64, stride=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.res_block3 = ResidualBlock(64, 128, stride=1)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, output_size)
        
    def forward(self, x):
        # Ensure input is in the right shape [batch, channels, freq, time]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.pool2(x)
        
        x = self.res_block2(x)
        x = self.pool3(x)
        
        x = self.res_block3(x)
        
        # Global pooling and flatten
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x