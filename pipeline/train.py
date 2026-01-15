"""
Training script for Vision Transformer models on retinal disease classification.

This script handles model selection, training loop, and evaluation using
PyTorch Lightning framework.

Author: Chidwipak Kuppani
"""

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import config
from core.utils import collate_fn, plot_results
from core.models.classifiers import SwinTransformer, DeiT, ResNet50, ViT

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=config.early_stopping_patience,
    strict=False,
    verbose=False,
    mode='min'
)

# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    mode='min',
    monitor='val_loss'
)

# CSV Logger for metrics
csv_logger = CSVLogger(
    save_dir='../outputs/', 
    name='training_logs', 
    flush_logs_every_n_steps=1
)

# Model selection based on config
MODEL_REGISTRY = {
    'SWIN': SwinTransformer,
    'VIT': ViT,
    'DeiT': DeiT,
    'ResNet': ResNet50,
}

if config.model_processor not in MODEL_REGISTRY:
    raise ValueError(
        f"Unknown model: {config.model_processor}. "
        f"Available models: {list(MODEL_REGISTRY.keys())}"
    )

model = MODEL_REGISTRY[config.model_processor]()
print(f"Selected model: {config.model_processor}")

# Initialize trainer
trainer = Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=config.num_epochs,
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=csv_logger,
    log_every_n_steps=5
)

# Train the model
trainer.fit(model)

# Evaluate on test set
print("=" * 50)
print("Testing...")
print("=" * 50)
trainer.test(model, ckpt_path='best')
print("=" * 50)

# Plot training results
plot_results()
