"""
Custom Dataset class for retinal image classification.

Handles loading of fundus images and multi-label annotations from CSV files.

Author: Chidwipak Kuppani
"""

import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from core import config


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for loading retinal fundus images with multi-label annotations.
    
    Args:
        csv_file: Path to CSV file containing image paths and labels
        root_dir: Root directory containing the images
        transform: Optional transform to apply to images
    """
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_names = ["N", "D", "C", "M"]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get image path based on dataset format
        if config.dataset == 'RFMID':
            img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx, 0]))
        else:
            img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        
        # Load image
        image = Image.open(img_name).convert('RGB')
        
        # Extract multi-label annotations
        label = []
        for name in self.label_names:
            label.append(1 if self.data_frame.at[idx, name] == 1 else 0)
        label = torch.tensor(label)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Convert to tensor
        image_tensor = torch.tensor(np.array(image))
        image_tensor = torch.permute(image_tensor, (2, 0, 1))

        return {
            "img": image, 
            "label": label, 
            "pixel_values": image_tensor
        }
