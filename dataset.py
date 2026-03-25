import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class KaggleNucleiDataset(Dataset):
    """
    PyTorch Dataset handle for the Kaggle 2018 Data Science Bowl (Nuclei Segmentation).
    Industry standard handling of paths, resizing, and tensor conversions.
    """
    def __init__(self, root_dir, image_size=256, transform=None):
        """
        Args:
            root_dir (string): Directory with all the image ID folders (e.g. data/kaggle_2018/stage1_train)
            image_size (int): Size to resize the images to (U-Net usually likes 256 or 512).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        
        # The Kaggle dataset contains one folder per image ID
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Please extract the data first.")
            
        self.image_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_folder = os.path.join(self.root_dir, img_id)
        
        # 1. Load the corresponding Image
        img_path = glob.glob(os.path.join(img_folder, 'images', '*.png'))[0]
        image = cv2.imread(img_path)
        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load and Combine all corresponding Masks
        mask_folder = os.path.join(img_folder, 'masks')
        mask_paths = glob.glob(os.path.join(mask_folder, '*.png'))
        
        # Initialize an empty mask of the same H x W as the image
        mask = np.zeros(image.shape[:2], dtype=np.bool_)
        
        for m_path in mask_paths:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            # Combine the current cell mask with the global mask via logical OR
            mask = np.maximum(mask, m > 0)
            
        # Convert boolean mask to float (0.0 and 1.0)
        mask = mask.astype(np.float32)
        
        # 3. Resize Image and Mask
        # U-Net expects fixed size inputs. Masks MUST use nearest neighbor to avoid blurring boundaries.
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # 4. Standardize for PyTorch: (C, H, W) instead of (H, W, C), and scale [0, 1]
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        # Add a channel dimension to the mask so shape is (1, H, W)
        mask = np.expand_dims(mask, axis=0)
        
        # Convert to Tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        
        # Apply optional custom transform logic (e.g. Albumentations) here later if needed
        if self.transform:
             pass 
             
        return image_tensor, mask_tensor
