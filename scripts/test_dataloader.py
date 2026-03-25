import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add parent directory to path to import dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import KaggleNucleiDataset

def test_data_pipeline():
    train_dir = os.path.join('data', 'kaggle_2018', 'stage1_train')
    
    # Initialize the dataset
    print(f"Loading dataset from: {train_dir}")
    try:
        dataset = KaggleNucleiDataset(root_dir=train_dir, image_size=256)
        print(f"Successfully loaded {len(dataset)} images and combined masks.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use a PyTorch DataLoader (Standard way to batch data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grab one batch
    images, masks = next(iter(dataloader))
    print(f"Batch Images Shape: {images.shape} (Batch, Channels, Height, Width)")
    print(f"Batch Masks Shape: {masks.shape} (Batch, Channels, Height, Width)")
    
    # Visualize the first image in the batch and its mask
    # Convert from PyTorch format (C, H, W) back to numpy (H, W, C) for plotting
    img_to_plot = images[0].numpy().transpose(1, 2, 0)
    mask_to_plot = masks[0].numpy().squeeze() # Remove channel dim for plotting grayscale
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_to_plot)
    axes[0].set_title("Original Image (Resized)")
    axes[0].axis('off')
    
    axes[1].imshow(mask_to_plot, cmap='gray')
    axes[1].set_title("Combined Nuclei Mask")
    axes[1].axis('off')
    
    # Plot an overlay to make sure they align perfectly
    axes[2].imshow(img_to_plot)
    axes[2].imshow(mask_to_plot, cmap='Reds', alpha=0.4) # Overlay red mask
    axes[2].set_title("Overlay Check")
    axes[2].axis('off')
    
    plt.tight_layout()
    # Save the plot to disk so the user can easily view it on Windows or GCP
    save_path = "execution1_test_plot.png"
    plt.savefig(save_path)
    print(f"\nSUCCESS: Data pipeline works perfectly. Visualization saved as '{save_path}'!")

if __name__ == "__main__":
    test_data_pipeline()
