"""
Attention Map Visualization for Vision Transformer models.

Generates heatmap overlays showing where the model focuses when making predictions,
providing interpretability for retinal disease classification.

Author: Chidwipak Kuppani
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoImageProcessor
from src.models.classifiers import ViT

# Color scheme
HEATMAP_CMAP = 'magma'  # Purple-based colormap

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image processor for ViT
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
image_mean = processor.image_mean
image_std = processor.image_std
normalize = Normalize(mean=image_mean, std=image_std)


def load_model(checkpoint_path):
    """
    Load a trained ViT model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        
    Returns:
        Loaded model in evaluation mode
    """
    model = ViT.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=224):
    """
    Preprocess an image for model inference.
    
    Args:
        image_path: Path to the input image
        image_size: Target size for the image
        
    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    transforms = Compose([
        Resize((image_size, image_size)),
        CenterCrop((image_size, image_size)),
        ToTensor(),
        normalize
    ])
    image_tensor = transforms(image).unsqueeze(0)
    return image_tensor.to(device)


def infer_attention_maps(model, image_tensor):
    """
    Extract attention maps from the model.
    
    Args:
        model: The ViT model
        image_tensor: Preprocessed input image
        
    Returns:
        Attention maps from the model
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        attentions = outputs[1]
    return attentions


def overlay_heatmap(image, heatmap, alpha=0.7, cmap=HEATMAP_CMAP):
    """
    Overlay attention heatmap on the original image.
    
    Args:
        image: Original image as numpy array
        heatmap: Attention heatmap
        alpha: Blending factor
        cmap: Matplotlib colormap name
    """
    # Resize heatmap to match image dimensions
    resized_heatmap = np.array(
        Image.fromarray(heatmap).resize((image.shape[1], image.shape[0]))
    )

    # Normalize heatmap
    heatmap_normalized = (resized_heatmap - resized_heatmap.min()) / \
                         (resized_heatmap.max() - resized_heatmap.min())

    # Apply colormap
    heatmap_colored = plt.get_cmap(cmap)(heatmap_normalized)

    # Blend with original image
    overlaid_image = (alpha * heatmap_colored[:, :, :3] + (1 - alpha) * image / 255.0)
    overlaid_image = np.clip(overlaid_image, 0, 1)

    # Create visualization
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_image)
    plt.title('Attention Map Overlay', fontsize=14)
    plt.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1)
    plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_attention(image_path, checkpoint_path):
    """
    Generate and display attention map for a given image.
    
    Args:
        image_path: Path to the input image
        checkpoint_path: Path to the model checkpoint
    """
    model = load_model(checkpoint_path)
    image_tensor = preprocess_image(image_path)
    attentions = infer_attention_maps(model, image_tensor)

    # Extract CLS token attention from last layer
    last_attention = attentions[-1]
    cls_attn_map = last_attention[:, :, 0, 1:].mean(dim=1).view(14, 14).cpu().numpy()
    cls_attn_map_normalized = (cls_attn_map - cls_attn_map.min()) / \
                              (cls_attn_map.max() - cls_attn_map.min())

    original_image = Image.open(image_path).convert('RGB')
    original_image_np = np.asarray(original_image)

    overlay_heatmap(original_image_np, cls_attn_map_normalized)


# Example usage (update paths as needed)
if __name__ == "__main__":
    # Configuration - update these paths for your setup
    DATA_DIR = "data/ODIR/images/"
    CHECKPOINT_PATH = "logs/experiment/version_0/checkpoints/best.ckpt"
    
    # Example image files to visualize
    example_images = [
        '16_right.jpg',
        '1164_left.jpg',
    ]
    
    for img_file in example_images:
        image_path = os.path.join(DATA_DIR, img_file)
        if os.path.exists(image_path):
            print(f"Generating attention map for: {img_file}")
            visualize_attention(image_path, CHECKPOINT_PATH)
        else:
            print(f"Image not found: {image_path}")
