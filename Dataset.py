import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from preprocess import Preprocessing

class MammogramData(Dataset):
    """ Dataset for Loading Prior and Current Mammograms with Current Mask """
    def __init__(self, prior_image_dir, current_image_dir, mask_dir, transform=None):
        self.prior_image_dir = prior_image_dir
        self.current_image_dir = current_image_dir
        self.mask_dir = mask_dir
        self.prior_filenames = sorted(os.listdir(prior_image_dir))
        self.current_filenames = sorted(os.listdir(current_image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform if transform else Preprocessing()
    
    def __len__(self):
        return len(self.prior_filenames)
    
    def __getitem__(self, idx):
        prior_path = os.path.join(self.prior_image_dir, self.prior_filenames[idx])
        current_path = os.path.join(self.current_image_dir, self.current_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        prior_image = cv2.imread(prior_path)
        current_image = cv2.imread(current_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if prior_image is None or current_image is None or mask is None:
            raise ValueError(f"Error loading images or mask: {prior_path}, {current_path}, {mask_path}")
        
        prior_image = self.transform.preprocess(prior_image)
        current_image = self.transform.preprocess(current_image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize mask to [0,1]
        
        return prior_image, current_image, mask

# Function to create DataLoader
def get_dataloader(prior_image_dir, current_image_dir, mask_dir, batch_size=16, shuffle=True, num_workers=4):
    dataset = MammogramData(prior_image_dir, current_image_dir, mask_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)