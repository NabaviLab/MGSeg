import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Dice Coefficient
def dice_coefficient(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# IoU (Intersection over Union)
def iou(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Save Segmentation Results
def save_mask(pred_mask, idx, output_dir="results/"):
    pred_mask = torch.sigmoid(pred_mask).cpu().numpy().squeeze()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}mask_{idx}.png", pred_mask)
    print(f"Saved: {output_dir}mask_{idx}.png")