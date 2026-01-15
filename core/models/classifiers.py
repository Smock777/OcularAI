"""
Vision Transformer Classifiers for Retinal Disease Detection

This module implements various pre-trained Vision Transformer and CNN architectures
for multi-label classification of retinal diseases using the ODIR-5K dataset.

Author: Chidwipak Kuppani
"""

import torch
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
from torchmetrics.classification import MultilabelRankingAveragePrecision

import pytorch_lightning as pl
from transformers import (
    AutoModelForImageClassification,
    DeiTForImageClassification,
    ResNetForImageClassification,
    ViTForImageClassification
)
from torch.utils.data import DataLoader
from torch import nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import config
from core.utils import load_data, collate_fn


class SwinTransformer(pl.LightningModule):
    """
    Swin Transformer model for multi-label retinal disease classification.
    Uses microsoft/swin-tiny-patch4-window7-224 pre-trained weights.
    """
    
    def __init__(self):
        super(SwinTransformer, self).__init__()

        # Load pre-trained Swin Transformer
        self.model = AutoModelForImageClassification.from_pretrained(
            'microsoft/swin-tiny-patch4-window7-224',
            num_labels=config.num_labels,
            problem_type="multi_label_classification",
            id2label=config.id2label,
            label2id=config.label2id,
            ignore_mismatched_sizes=True
        )

        # Initialize metric tracking lists
        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []

        # Load datasets
        self.train_ds, self.val_ds, self.test_ds = load_data(dataset=config.dataset)

        # Custom classification head
        custom_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_labels)
        )
        self.model.classifier = custom_head

        # Freeze backbone, train only classifier
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        """Common forward pass for train/val/test steps."""
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, shuffle=True, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def process_epoch_end(self, predictions, labels, phase):
        """Calculate and log metrics at epoch end."""
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)
        f1_macro = f1_score(preds, lbls, average='macro')
        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase}_multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i}_{phase}_f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")


class DeiT(pl.LightningModule):
    """
    DeiT (Data-efficient Image Transformer) for multi-label retinal disease classification.
    Uses facebook/deit-base-distilled-patch16-224 pre-trained weights.
    """
    
    def __init__(self):
        super(DeiT, self).__init__()

        # Load pre-trained DeiT model
        self.model = DeiTForImageClassification.from_pretrained(
            'facebook/deit-base-distilled-patch16-224',
            num_labels=config.num_labels,
            problem_type="multi_label_classification",
            id2label=config.id2label,
            label2id=config.label2id,
            ignore_mismatched_sizes=True
        )

        # Initialize metric tracking lists
        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []

        # Load datasets
        self.train_ds, self.val_ds, self.test_ds = load_data(dataset=config.dataset)

        # Custom classification head
        custom_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_labels),
        )
        self.model.classifier = custom_head

        # Freeze backbone, train only classifier
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        """Common forward pass for train/val/test steps."""
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, shuffle=True, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def process_epoch_end(self, predictions, labels, phase):
        """Calculate and log metrics at epoch end."""
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)
        f1_macro = f1_score(preds, lbls, average='macro')
        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase}_multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            self.log(f"{i}_{phase}_f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")


class ResNet50(pl.LightningModule):
    """
    ResNet-50 CNN baseline for multi-label retinal disease classification.
    Uses microsoft/resnet-50 pre-trained weights.
    """
    
    def __init__(self):
        super(ResNet50, self).__init__()

        # Load pre-trained ResNet-50
        self.model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=config.num_labels,
            problem_type="multi_label_classification",
            id2label=config.id2label,
            label2id=config.label2id,
            ignore_mismatched_sizes=True
        )

        # Initialize metric tracking lists
        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []

        # Load datasets
        self.train_ds, self.val_ds, self.test_ds = load_data(dataset=config.dataset)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        """Common forward pass for train/val/test steps."""
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, shuffle=True, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def process_epoch_end(self, predictions, labels, phase):
        """Calculate and log metrics at epoch end."""
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)
        f1_macro = f1_score(preds, lbls, average='macro')
        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase}_multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            self.log(f"{i}_{phase}_f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")


class ViT(pl.LightningModule):
    """
    Vision Transformer (ViT) for multi-label retinal disease classification.
    Uses google/vit-base-patch16-224 pre-trained weights.
    """
    
    def __init__(self):
        super(ViT, self).__init__()

        # Load pre-trained ViT model
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=config.num_labels,
            problem_type="multi_label_classification",
            id2label=config.id2label,
            label2id=config.label2id,
            ignore_mismatched_sizes=True
        )

        # Initialize metric tracking lists
        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []

        # Load datasets
        self.train_ds, self.val_ds, self.test_ds = load_data(dataset=config.dataset)

        # Custom classification head with BatchNorm
        custom_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, config.num_labels)
        )
        self.model.classifier = custom_head

        # Freeze backbone, train only classifier
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        """Common forward pass for train/val/test steps."""
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, shuffle=True, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, collate_fn=collate_fn,
            batch_size=config.batch_size, num_workers=4,
            persistent_workers=True, pin_memory=True
        )

    def process_epoch_end(self, predictions, labels, phase):
        """Calculate and log metrics at epoch end."""
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)
        f1_macro = f1_score(preds, lbls, average='macro')
        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase}_multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            self.log(f"{i}_{phase}_f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")
