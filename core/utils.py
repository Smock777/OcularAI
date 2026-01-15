"""
Utility functions for data loading, preprocessing, and visualization.

Author: Chidwipak Kuppani
"""

import pandas as pd
from core.data.dataset import RetinalDataset
from torchvision.transforms import v2

from transformers import AutoImageProcessor

import torch
import matplotlib.pyplot as plt
import os
import glob

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core import config

# Color scheme for plots (teal and purple theme)
TRAIN_COLOR = '#00897B'  # Teal
VAL_COLOR = '#7B1FA2'    # Purple


def plot_results():
    """
    Plot training and validation F1 scores from the latest experiment.
    Reads metrics from the most recent experiment log directory.
    """
    base_path = 'outputs/training_logs/'

    subdirs = glob.glob(os.path.join(base_path, '*/'))

    if not subdirs:
        print("No subdirectories found in the training_logs directory.")
        return

    latest_dir = max(subdirs, key=os.path.getctime)
    metrics_file = os.path.join(latest_dir, 'metrics.csv')

    if not os.path.isfile(metrics_file):
        print(f"No metrics.csv file found in: {latest_dir}")
        return
    
    print(f"Plotting results from {metrics_file}")

    # Read metrics and create subplots
    metrics_df = pd.read_csv(metrics_file)
    fig, axs = plt.subplots(1, config.num_labels, figsize=(16, 8))

    for i in range(config.num_labels):
        label_train_f1 = metrics_df.groupby('epoch')[f'{i}_train_f1'].mean()
        label_val_f1 = metrics_df.groupby('epoch')[f'{i}_val_f1'].mean()
        epochs = range(1, len(label_train_f1) + 1)

        axs[i].plot(epochs, label_train_f1, color=TRAIN_COLOR, 
                    label=f'Train F1', linewidth=2)
        axs[i].plot(epochs, label_val_f1, color=VAL_COLOR, 
                    label=f'Val F1', linewidth=2)
        axs[i].set_title(f'Label {config.id2label[i]} - F1 Score', fontsize=12)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('F1 Score')
        axs[i].legend()
        axs[i].set_ylim(0, 1)
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_results.png', dpi=150)
    plt.show()


# Image processor selection based on model
PROCESSOR_REGISTRY = {
    'SWIN': "microsoft/swin-tiny-patch4-window7-224",
    'VIT': "google/vit-base-patch16-224",
    'DeiT': "facebook/deit-base-distilled-patch16-224",
    'ResNet': "microsoft/resnet-50",
}

processor_name = PROCESSOR_REGISTRY.get(config.model_processor, "google/vit-base-patch16-224")
processor = AutoImageProcessor.from_pretrained(processor_name)

image_mean = processor.image_mean
image_std = processor.image_std

# Data normalization transform
normalize = v2.Normalize(mean=image_mean, std=image_std)

# Image size configuration
IMAGE_SIZE = (224, 224)


def load_data(dataset):
    """
    Load train, validation, and test datasets with appropriate transforms.
    
    Args:
        dataset: Dataset name ('ODIR' or 'RFMID')
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    if dataset == 'ODIR':
        path_train = 'data/ODIR/train_strat.csv'
        path_val = 'data/ODIR/val_strat.csv'
        path_test = 'data/ODIR/test_strat.csv'
        root_dir = 'data/ODIR/images/'
    elif dataset == 'RFMID':
        path_train = './data/RFMiD/RFMiD_Training_Labels_curated.csv'
        path_val = './data/RFMiD/RFMiD_Validation_Labels_curated.csv'
        path_test = './data/RFMiD/RFMiD_Testing_Labels_curated.csv'
        root_dir = './data/RFMiD/images/'
    else:
        raise ValueError('Invalid dataset. Use ODIR or RFMID')

    # Training augmentations
    train_transforms = v2.Compose([
        v2.Resize(IMAGE_SIZE),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.1),
        v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
        v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),
        v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
        v2.ToTensor(),
        normalize
    ])

    # Validation/Test transforms (no augmentation)
    val_transforms = v2.Compose([
        v2.Resize(IMAGE_SIZE),
        v2.ToTensor(),
        normalize
    ])

    train_ds = RetinalDataset(csv_file=path_train, root_dir=root_dir, transform=train_transforms)
    val_ds = RetinalDataset(csv_file=path_val, root_dir=root_dir, transform=val_transforms)
    test_ds = RetinalDataset(csv_file=path_test, root_dir=root_dir, transform=val_transforms)

    return train_ds, val_ds, test_ds


def collate_fn(examples):
    """
    Custom collate function for DataLoader.
    Stacks pixel values and labels into batches.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = torch.permute(pixel_values, (0, 2, 1, 3))
    labels = torch.stack([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels": labels}
