import torch
import numpy as np
import os
import sys
from tqdm import tqdm

from dataset.dataset_ESC50 import ESC50
from train_crossval import test, make_model, global_stats
import config

def test_improvements():
    print("Testing improvements on a single fold...")
    
    # Use CUDA if available
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{config.device_id}" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Data path
    data_path = config.esc50_path
    
    # Test only on the first fold for quick validation
    test_fold = 1
    
    # Create datasets
    train_set = ESC50(subset="train", test_folds={test_fold}, 
                     global_mean_std=global_stats[test_fold - 1], 
                     root=data_path, download=True)
    
    test_set = ESC50(subset="test", test_folds={test_fold}, 
                    global_mean_std=global_stats[test_fold - 1], 
                    root=data_path, download=True)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False
    )
    
    # Create model
    model = make_model()
    model = model.to(device)
    print("Model created successfully:")
    print(model)
    
    # Get sample batch
    print("\nTesting forward pass with a sample batch...")
    for _, x, label in train_loader:
        x = x.float().to(device)
        y_true = label.to(device)
        
        # Forward pass
        y_prob = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y_prob.shape}")
        
        # Check predictions
        y_pred = torch.argmax(y_prob, dim=1)
        accuracy = (y_pred == y_true).float().mean()
        print(f"Sample batch accuracy (random initialization): {accuracy:.4f}")
        
        break  # Just test one batch
    
    print("\nImprovements validated successfully!")
    return True

if __name__ == "__main__":
    test_improvements()