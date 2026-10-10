import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import random

class SpecAugment(nn.Module):
    """
    Google Brain SpecAugment with stochastic application probability p=0.8:
    Applies Frequency Masking and Time Masking directly to Mel-Spectrogram tensors.
    """
    def __init__(self, freq_mask_param=16, time_mask_param=12, p=0.8):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.p = p

    def forward(self, x):
        # x shape: (1, 128, 63) or (B, 1, 128, 63)
        if random.random() < self.p:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x

class VoiceDataset(Dataset):
    """
    In-memory PyTorch Dataset for ultra-fast training without disk I/O bottlenecks.
    """
    def __init__(self, X, y, transform=None):
        self.X = X.float()
        self.y = y.long()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]

def get_dataloaders(dataset_path="MT/dataset_processed.pt" if os.path.exists("MT/dataset_processed.pt") else "ADL/dataset_processed.pt", batch_size=32, use_specaug=True):
    data = torch.load(dataset_path, weights_only=False)
    
    train_transform = SpecAugment(freq_mask_param=16, time_mask_param=12, p=0.8) if use_specaug else None

    train_dataset = VoiceDataset(data["X_train"], data["y_train"], transform=train_transform)
    val_dataset = VoiceDataset(data["X_val"], data["y_val"], transform=None)
    test_dataset = VoiceDataset(data["X_test"], data["y_test"], transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "label_map": data["label_map"],
        "class_names": data["class_names"],
        "mel_shape": data["mel_shape"],
    }

    return train_loader, val_loader, test_loader, metadata

if __name__ == "__main__":
    train_loader, val_loader, test_loader, meta = get_dataloaders()
    for batch_X, batch_y in train_loader:
        print(f"Train batch X: {batch_X.shape}, y: {batch_y.shape}")
        break
    print("[SUCCESS] dataset.py updated with p=0.8 SpecAugment!")