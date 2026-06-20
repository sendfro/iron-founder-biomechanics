import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==================================================
# 1. THE DATASET (Synthetic Knee Images)
# ==================================================
class SyntheticKneeDataset(Dataset):
    """
    Generates synthetic 64x64 RGB images to simulate cropped YOLO knee bounding boxes.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # Simulating a batch of 64x64 RGB images (3 channels)
        # Pixel values normalized between 0 and 1
        self.data = torch.rand((num_samples, 3, 64, 64), dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Autoencoders compare the input to itself, so X and Y are the exact same image
        image = self.data[idx]
        return image, image

# ==================================================
# 2. THE CONVOLUTIONAL AUTOENCODER ARCHITECTURE
# ==================================================
class KneeAnomalyAutoencoder(nn.Module):
    def __init__(self):
        super(KneeAnomalyAutoencoder, self).__init__()
        
        # ENCODER: Compresses the 64x64 knee image down to its core structural features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU()
        )
        
        # DECODER: Attempts to reconstruct the image from the compressed features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 64x64
            nn.Sigmoid() # Squashes output pixels to valid [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==================================================
# 3. THE TRAINING PIPELINE
# ==================================================
def train_autoencoder():
    print("Initializing Convolutional Autoencoder Pipeline...")
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 15
    
    dataset = SyntheticKneeDataset(num_samples=1500)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KneeAnomalyAutoencoder().to(device)
    
    # MSE Loss checks how perfectly the reconstructed image matches the original input image
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    print(f"Training initialized on engine target: {device}\n" + "="*40)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            reconstructed_images = model(batch_x)
            loss = criterion(reconstructed_images, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        total_epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Reconstruction Loss: {total_epoch_loss:.5f}")
        
    model_filename = "knee_autoencoder.pt"
    torch.save(model.state_dict(), model_filename)
    print("="*40 + f"\nTraining Complete! Autoencoder weights exported to '{model_filename}'")

if __name__ == "__main__":
    train_autoencoder()