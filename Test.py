import torch
from torch.utils.data import DataLoader
from dataset import get_dataloader
from encoder import TransformerEncoder
from decoder import Decoder
from fdb import FeatureDiscrepancyBlock
import cv2
import numpy as np

# Model Loading
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = TransformerEncoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
encoder.load_state_dict(torch.load("encoder.pth", map_location=DEVICE))
decoder.load_state_dict(torch.load("decoder.pth", map_location=DEVICE))
encoder.eval()
decoder.eval()

# Load Test Data
test_loader = get_dataloader("our_mammogram/prior_images", "our_mammogram/current_images", "our_mammogram/masks", batch_size=1, shuffle=False)

# Evaluation Function
def evaluate():
    with torch.no_grad():
        for idx, (prior_image, current_image, mask) in enumerate(test_loader):
            prior_image, current_image = prior_image.to(DEVICE), current_image.to(DEVICE)
            
            prior_features = encoder(prior_image)
            current_features = encoder(current_image)
            
            fdb_outputs = []
            prev_fdb = None
            for i in range(len(prior_features)):
                fdb_block = FeatureDiscrepancyBlock(prior_features[i].shape[1]).to(DEVICE)
                fdb_output = fdb_block(prior_features[i], current_features[i], prev_fdb)
                fdb_outputs.append(fdb_output)
                prev_fdb = fdb_output
            
            predicted_mask = decoder(fdb_outputs)
            predicted_mask = torch.sigmoid(predicted_mask).cpu().numpy().squeeze()
            
            mask = mask.cpu().numpy().squeeze()
            
            cv2.imwrite(f"results/mask_{idx}.png", (predicted_mask * 255).astype(np.uint8))
            print(f"Saved: results/mask_{idx}.png")

if __name__ == "__main__":
    evaluate()