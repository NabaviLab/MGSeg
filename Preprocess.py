import cv2
import numpy as np
import torch
from torchvision import transforms

class Preprocessing:
    """ Preprocessing Pipeline for Mammogram Images """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def artifact_removal(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(img, img, mask=mask)
    
    def apply_CLAHE(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img
    
    def preprocess(self, img):
        img = self.artifact_removal(img)
        img = self.apply_CLAHE(img)
        img = self.transform(img)
        return img