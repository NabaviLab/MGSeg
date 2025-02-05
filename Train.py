import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_dataloader
from encoder import TransformerEncoder
from decoder import Decoder
from fdb import FeatureDiscrepancyBlock
from loss import CompositeLoss

# Training Parameters
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model
encoder = TransformerEncoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
loss_function = CompositeLoss().to(DEVICE)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# Load Data
data_loader = get_dataloader("our_mammogram/prior_images", "our_mammogram/current_images", "our_mammogram/masks", batch_size=BATCH_SIZE)

# Training Loop
def train():
    encoder.train()
    decoder.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for prior_image, current_image, mask in data_loader:
            prior_image, current_image, mask = prior_image.to(DEVICE), current_image.to(DEVICE), mask.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            prior_features = encoder(prior_image)
            current_features = encoder(current_image)
            
            # Feature Discrepancy Computation
            fdb_outputs = []
            prev_fdb = None
            for i in range(len(prior_features)):
                fdb_block = FeatureDiscrepancyBlock(prior_features[i].shape[1]).to(DEVICE)
                fdb_output = fdb_block(prior_features[i], current_features[i], prev_fdb)
                fdb_outputs.append(fdb_output)
                prev_fdb = fdb_output
            
            # Decode to Segmentation Map
            predicted_mask = decoder(fdb_outputs)
            
            # Compute Loss
            loss = loss_function(predicted_mask, mask, encoder, epoch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(data_loader):.4f}")

if __name__ == "__main__":
    train()