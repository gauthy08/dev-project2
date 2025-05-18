import torch
import numpy as np

class SpecAugment:
    def __init__(self, freq_mask_param=10, time_mask_param=20, n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
    def __call__(self, spec):
        """
        GPU-optimierte Version von SpecAugment
        spec: [1, n_mels, time_steps] mel spectrogram
        """
        # Sicherstellen, dass spec ein Tensor ist
        if not isinstance(spec, torch.Tensor):
            spec = torch.tensor(spec, dtype=torch.float)
            
        # Kopieren, um das Original nicht zu verändern
        spec = spec.clone()
        
        device = spec.device
        
        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, spec.shape[1] - f)
            spec[0, f0:f0+f, :] = 0
            
        # Apply time masks
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, spec.shape[2] - t)
            spec[0, :, t0:t0+t] = 0
            
        return spec