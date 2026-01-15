# 📚 OcularAI: Complete Project Documentation

## Table of Contents
1. [Project Motivation](#project-motivation)
2. [Development Timeline](#development-timeline)
3. [Dataset & Preprocessing](#dataset--preprocessing)
4. [Model Architectures](#model-architectures)
5. [Training Pipeline](#training-pipeline)
6. [Results & Analysis](#results--analysis)
7. [Tech Stack Justification](#tech-stack-justification)
8. [References](#references)

---

## 1. Project Motivation

### 💡 The Problem

Retinal diseases are leading causes of preventable blindness worldwide:
- **Diabetic Retinopathy:** Affects 1 in 3 diabetics, leading cause of blindness in working-age adults
- **Cataracts:** Responsible for 51% of world blindness (WHO)
- **Pathological Myopia:** Growing prevalence with increased screen time

**Clinical Challenge:**
- Manual fundus image analysis is time-consuming and requires specialized ophthalmologists
- Rural areas lack access to eye specialists
- Early detection is crucial but often missed

### 🎯 Research Question

**Can Vision Transformers outperform traditional CNNs for multi-label retinal disease classification, and which architecture provides the best balance of accuracy and interpretability?**

### 🌟 Why Vision Transformers?

Traditional CNNs have limitations for medical imaging:
- ❌ Limited receptive field (local features only)
- ❌ Fixed grid structure
- ❌ Poor interpretability (black box)

**Vision Transformer Advantages:**
- ✅ Global attention mechanism (sees entire image)
- ✅ Flexible patch-based processing
- ✅ Attention maps provide interpretability
- ✅ Transfer learning from large-scale pre-training

### 🔬 Comparison Study

This project compares 4 architectures:
1. **Swin Transformer** - Microsoft's shifted window attention
2. **DeiT** - Facebook's data-efficient ViT
3. **ViT** - Google's original Vision Transformer
4. **ResNet-50** - CNN baseline for comparison

---

## 2. Development Timeline

### 📅 July 2025 - 4 Week Research Sprint

#### **Week 1: Research & Dataset Preparation** (July 1-7, 2025)

**Day 1-2: Literature Review**
- Studied Vision Transformer papers (ViT, DeiT, Swin)
- Reviewed retinal disease classification benchmarks
- Analyzed ODIR-5K dataset characteristics

**Day 3-4: Dataset Acquisition & Analysis**
- Downloaded ODIR-5K from Kaggle
- Performed exploratory data analysis (`pipeline/analysis/eda.py`)
- Identified class imbalance issue

**Day 5-6: Data Preprocessing**
- Quality filtering (removed low-quality images)
- Stratified train/val/test split (70/15/15)
- Image preprocessing pipeline (resize, normalize)

**Day 7: Multi-label Strategy**
- Decided on 4 primary classes: N, D, C, M
- Implemented weighted BCE loss for class imbalance
- Created dataset statistics report

**Deliverables:**
- Cleaned dataset splits (train_strat.csv, val_strat.csv, test_strat.csv)
- EDA visualizations
- Dataset statistics

---

#### **Week 2: Model Architecture Implementation** (July 8-14, 2025)

**Phase 2.1: Base Architecture Design** (Days 8-9)

Created modular classifier design (`core/models/classifiers.py`):
```python
# All models share common interface via PyTorch Lightning
class BaseClassifier(pl.LightningModule):
    - forward(): feature extraction → classification head
    - common_step(): unified train/val/test logic
    - configure_optimizers(): AdamW with weight decay
    - process_epoch_end(): metrics computation
```

**Phase 2.2: Swin Transformer** (Day 10)

```python
# microsoft/swin-tiny-patch4-window7-224
class SwinTransformer(pl.LightningModule):
    - Shifted window attention for efficiency
    - Hierarchical feature maps
    - Custom classification head (768 → 4)
```

**Key Implementation Details:**
- Input: 224×224 images
- Patch size: 4×4
- Window size: 7×7
- Hidden dim: 768

**Phase 2.3: DeiT Implementation** (Day 11)

```python
# facebook/deit-base-distilled-patch16-224
class DeiT(pl.LightningModule):
    - Knowledge distillation from larger teacher
    - Data-efficient training
    - Distillation token output
```

**Phase 2.4: ViT & ResNet Baseline** (Days 12-13)

```python
# google/vit-base-patch16-224
class ViT(pl.LightningModule):
    - Original vision transformer
    - Standard 16×16 patches
    
# microsoft/resnet-50
class ResNet50(pl.LightningModule):
    - CNN baseline for comparison
    - Pre-trained ImageNet weights
```

**Phase 2.5: Unified Training Interface** (Day 14)

```python
# Model registry for easy switching
MODEL_REGISTRY = {
    'SWIN': SwinTransformer,
    'VIT': ViT,
    'DeiT': DeiT,
    'ResNet': ResNet50,
}
```

---

#### **Week 3: Training & Experimentation** (July 15-21, 2025)

**Phase 3.1: Training Infrastructure** (Days 15-16)

**PyTorch Lightning Setup:**
```python
trainer = Trainer(
    accelerator='gpu',          # RTX 6000
    max_epochs=400,             # With early stopping
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(save_top_k=1)
    ],
    logger=CSVLogger()
)
```

**Hardware Configuration:**
- GPU: NVIDIA RTX 6000 (24GB VRAM)
- RAM: 128GB
- Storage: NVMe SSD
- CUDA: 12.0
- PyTorch: 2.2.0

**Phase 3.2: Training Experiments** (Days 17-19)

**Hyperparameter Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Better generalization than Adam |
| Learning Rate | 5×10⁻⁴ | Standard for fine-tuning ViTs |
| Weight Decay | 0.01 | Regularization |
| Batch Size | 16 | GPU memory constraint |
| Max Epochs | 400 | With early stopping |
| Patience | 10 | Epochs without improvement |

**Loss Function:**
```python
# Weighted BCE for class imbalance
pos_weight = compute_class_weights(train_labels)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Training Results:**
```
Swin Transformer:
  - Best Epoch: 87
  - Training Time: 4.2 hours
  - Final Val Loss: 0.312
  
DeiT:
  - Best Epoch: 92
  - Training Time: 5.1 hours
  - Final Val Loss: 0.298
  
ViT:
  - Best Epoch: 78
  - Training Time: 4.8 hours
  - Final Val Loss: 0.334
  
ResNet-50:
  - Best Epoch: 65
  - Training Time: 2.3 hours
  - Final Val Loss: 0.389
```

**Phase 3.3: Logging & Monitoring** (Day 20-21)

Implemented comprehensive logging:
- Training/validation loss curves
- Per-class F1 scores
- Confusion matrices
- Learning rate scheduling

---

#### **Week 4: Evaluation & Visualization** (July 22-28, 2025)

**Phase 4.1: Test Set Evaluation** (Days 22-23)

**Final Results:**
| Model | Normal | DR | Cataract | Myopia | F1 Macro | Ranking AP |
|-------|--------|-----|----------|--------|----------|------------|
| Swin | **80.8%** | 61.1% | **86.3%** | 95.9% | **81.1%** | **81.0%** |
| DeiT | 78.9% | **63.5%** | 85.1% | **98.0%** | 81.4% | 80.8% |
| ViT | 76.8% | 63.4% | 83.3% | 95.9% | 79.9% | 78.6% |
| ResNet | 80.1% | 53.7% | 79.1% | **98.0%** | 77.7% | 79.4% |

**Key Findings:**
1. Vision Transformers outperform CNN baseline (+3.4% F1 Macro vs ResNet)
2. Swin achieves best overall balance
3. Diabetic Retinopathy is most challenging class
4. Myopia detection is highly accurate (>95% all models)

**Phase 4.2: Attention Visualization** (Days 24-25)

```python
# pipeline/visualization/attention.py
def visualize_attention(model, image, layer=-1):
    # Extract attention weights
    attentions = model.get_attention_weights(image)
    
    # Average across heads
    attention_map = attentions.mean(dim=1)
    
    # Overlay on original image
    heatmap = generate_heatmap(attention_map, image)
    return heatmap
```

**Observations:**
- Models correctly focus on relevant pathology regions
- Swin shows more localized attention patterns
- ViT has more distributed attention

**Phase 4.3: Documentation & Cleanup** (Days 26-28)

- Wrote comprehensive README
- Created project structure documentation
- Added requirements.txt
- Cleaned up experimental code

---

## 3. Dataset & Preprocessing

### ODIR-5K Dataset

**Source:** Peking University / Shanggong Medical Technology
**Competition:** ODIR-2019 Challenge

**Characteristics:**
- 5,000 patient records
- Paired left/right fundus images
- 8 diagnostic categories (reduced to 4 for this project)

### Class Selection

| Original Label | Description | Selected |
|----------------|-------------|----------|
| N | Normal | ✅ |
| D | Diabetic Retinopathy | ✅ |
| G | Glaucoma | ❌ (low samples) |
| C | Cataract | ✅ |
| A | AMD | ❌ (low samples) |
| H | Hypertension | ❌ (low samples) |
| M | Pathological Myopia | ✅ |
| O | Other | ❌ (ambiguous) |

### Data Split

| Split | Normal | DR | Cataract | Myopia | Total |
|-------|--------|-----|----------|--------|-------|
| Train | 1337 | 1079 | 195 | 163 | 2774 |
| Val | 325 | 285 | 49 | 39 | 698 |
| Test | 402 | 354 | 58 | 53 | 867 |

### Preprocessing Pipeline

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])
```

**No augmentation used** to preserve medical image integrity.

---

## 4. Model Architectures

### Swin Transformer (Best Performer)

**Pre-trained:** `microsoft/swin-tiny-patch4-window7-224`

**Architecture:**
```
Input (224×224×3)
    │
    ▼
Patch Partition (56×56×96)
    │
    ▼
Stage 1: 2× Swin Blocks (56×56×96)
    │
    ▼
Patch Merging (28×28×192)
    │
    ▼
Stage 2: 2× Swin Blocks (28×28×192)
    │
    ▼
Patch Merging (14×14×384)
    │
    ▼
Stage 3: 6× Swin Blocks (14×14×384)
    │
    ▼
Patch Merging (7×7×768)
    │
    ▼
Stage 4: 2× Swin Blocks (7×7×768)
    │
    ▼
Global Average Pool → FC(768→4) → Sigmoid
```

**Key Innovation:** Shifted window attention for O(n) complexity.

### DeiT (Data-Efficient)

**Pre-trained:** `facebook/deit-base-distilled-patch16-224`

**Key Features:**
- Knowledge distillation from larger teacher
- Distillation token in addition to CLS token
- Data augmentation strategies (RandAugment, Mixup)

### ViT (Original)

**Pre-trained:** `google/vit-base-patch16-224`

**Architecture:**
- 16×16 patches (14×14 = 196 tokens)
- 12 transformer blocks
- 768 hidden dimension
- 12 attention heads

### ResNet-50 (Baseline)

**Pre-trained:** `microsoft/resnet-50`

**Purpose:** CNN baseline for comparison with Vision Transformers.

---

## 5. Training Pipeline

### PyTorch Lightning Framework

**Benefits:**
- Separation of research code from engineering code
- Automatic GPU handling
- Built-in logging and checkpointing
- Easy distributed training

### Training Loop

```python
# Simplified training loop
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        images, labels = batch
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss, metrics = validate(model, val_loader)
    
    # Early stopping check
    if val_loss < best_loss:
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### Metrics

**Primary Metrics:**
- **F1 Macro:** Average F1 across all classes
- **Ranking Average Precision:** Multi-label ranking metric

**Per-Class Metrics:**
- Precision, Recall, F1 for each disease

---

## 6. Results & Analysis

### Performance Summary

**Ranking by F1 Macro:**
1. 🥇 **DeiT:** 81.4%
2. 🥈 **Swin:** 81.1%
3. 🥉 **ViT:** 79.9%
4. 4th **ResNet-50:** 77.7%

**Ranking by Ranking AP:**
1. 🥇 **Swin:** 81.0%
2. 🥈 **DeiT:** 80.8%
3. 🥉 **ResNet-50:** 79.4%
4. 4th **ViT:** 78.6%

### Key Insights

1. **Vision Transformers > CNNs**
   - All ViT variants outperform ResNet-50 baseline
   - Global attention captures inter-region dependencies

2. **Swin's Efficiency**
   - Best balance of accuracy and computational cost
   - Shifted windows enable linear complexity

3. **DeiT's Distillation**
   - Knowledge distillation improves generalization
   - Best per-class performance on challenging DR class

4. **Class Difficulty**
   - **Easy:** Myopia (>95% F1) - distinctive features
   - **Hard:** Diabetic Retinopathy (53-63% F1) - subtle microaneurysms

---

## 7. Tech Stack Justification

### PyTorch Lightning
**Why:**
- ✅ Clean separation of model/training code
- ✅ Built-in best practices
- ✅ Easy experiment tracking
- ✅ Automatic mixed precision support

### HuggingFace Transformers
**Why:**
- ✅ Pre-trained Vision Transformer models
- ✅ Unified API across architectures
- ✅ Easy fine-tuning
- ✅ Active community

### ODIR-5K Dataset
**Why:**
- ✅ Standard benchmark in medical imaging
- ✅ Multi-label annotations
- ✅ Sufficient size for fine-tuning
- ✅ Real clinical data

---

## 8. References

### Academic Papers

1. **Dosovitskiy et al. (2021)**
   "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   *ICLR 2021*

2. **Liu et al. (2021)**
   "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
   *ICCV 2021*

3. **Touvron et al. (2021)**
   "Training data-efficient image transformers & distillation through attention"
   *ICML 2021*

4. **He et al. (2016)**
   "Deep Residual Learning for Image Recognition"
   *CVPR 2016*

### Dataset

**ODIR-5K**
- Peking University / Shanggong Medical Technology
- https://odir2019.grand-challenge.org/

### Libraries

- PyTorch 2.2.0
- PyTorch Lightning 2.2.0
- HuggingFace Transformers 4.39.0
- NumPy, Pandas, Matplotlib

---

## Development Environment

**Remote SSH System:**
- **GPU:** NVIDIA RTX 6000 (24GB VRAM)
- **CPU:** Intel Xeon (32 cores)
- **RAM:** 128GB
- **Storage:** 1TB NVMe SSD
- **OS:** Ubuntu 22.04 LTS
- **CUDA:** 12.0
- **Python:** 3.12

---

## Lessons Learned

### What Went Well ✅
- Vision Transformers showed clear advantage over CNN
- PyTorch Lightning made experimentation efficient
- Pre-trained models transferred well to medical domain

### Challenges Faced ⚠️
- Class imbalance required careful loss weighting
- DR class remained challenging (subtle features)
- Training time (~5 hours per model)

### Future Improvements 🚀
- Ensemble multiple architectures
- Add more disease classes
- Implement cross-attention for paired eye analysis
- Deploy as clinical decision support tool

---

**Last Updated:** January 15, 2026  
**Project Duration:** July 2025 (4 weeks)  
**Total Lines of Code:** ~1,500  
**Models Trained:** 4  
**Best F1 Macro:** 81.4% (DeiT)
