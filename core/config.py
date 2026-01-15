"""
Configuration file for Vision Transformer experiments on retinal disease classification.

Author: Chidwipak Kuppani
"""

import torch.nn as nn
import torch

# Dataset Configuration
dataset = 'ODIR'

# Label Mappings
# N: Normal, D: Diabetic Retinopathy, C: Cataract, M: Myopia
id2label = {0: "N", 1: "D", 2: "C", 3: "M"}
label2id = {label: id for id, label in id2label.items()}

# Classification Setup
num_labels = 4
class_weights = torch.tensor(
    [1.03297754e-05, 1.27997310e-05, 7.08251781e-05, 8.47295075e-05], 
    dtype=torch.float
).to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()

# Training Hyperparameters
learning_rate = 0.0005
weight_decay = 0.0
batch_size = 128

# Early Stopping Configuration
early_stopping_patience = 10
num_epochs = 400

# Model Selection
# Options: SWIN, VIT, DeiT, ResNet
model_processor = 'VIT'
