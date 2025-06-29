# data.py
from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision.transforms as T

# Provided normalization statistics
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std  = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str):
        """
        data: pandas DataFrame loaded from data.csv with columns
              ['path','crack','inactive'].
        mode: 'train' or 'val' (not used for splitting here, but available for
              future augmentations).
        """
        self.data = data
        # minimal transform chain: to PIL → resize → to tensor → normalize
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((300, 300)),
            T.ToTensor(),
            T.Normalize(mean=train_mean, std=train_std),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Load the image
        img = imread(row['path'])
        # If grayscale, convert to RGB
        if img.ndim == 2:
            img = gray2rgb(img)
        # Ensure uint8 for PIL
        img = img.astype(np.uint8)
        # Apply transforms
        img_t = self.transform(img)
        # Multi-label tensor [crack, inactive]
        label = torch.tensor([row['crack'], row['inactive']], dtype=torch.float32)
        return img_t, label
