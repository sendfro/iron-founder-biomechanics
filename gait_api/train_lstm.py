import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==================================================
# 1. THE DATASET ARCHITECTURE (Synthetic Data)
# ==================================================
class BiomechanicsDataset(Dataset):
    """
    Custom Dataset to handle sequential movement data.
    Input (X): Shape (sequence_length, 34) -> 17 keypoints * 2 coordinates (x, y)
    Target (Y): Shape (2) -> [Peak GRF, Lateral Knee Load Multiplier]
    """
    def __init__(self, num_samples=500, seq_length=60):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        self.X_data = []
        self.Y_data = []
        
        for _ in range(num_samples):
            frames = []
            base_pose = np.random.uniform(100, 500, size=(17, 2))
            
            valgus_tendency = np.random.uniform(0.1, 1.5)
            for t in range(seq_length):
                movement_offset = np.sin((t / seq_length) * np.pi) * 50
                current_frame = base_pose.copy()
                current_frame[13] += [movement_offset * valgus_tendency, 0]
                current_frame[14] -= [movement_offset * valgus_tendency, 0]
                frames.append(current_frame.flatten()) 
                
            self.X_data.append(frames)
            
            peak_grf = 1.5 + (valgus_tendency * 0.5) + np.random.normal(0, 0.1)
            lateral_load = 0.3 + (valgus_tendency * 0.6) + np.random.normal(0, 0.05)
            self.Y_data.append([peak_grf, lateral_load])

        self.X_data = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        self.Y_data = torch.tensor(np.array(self.Y_data), dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X_data[idx], self.Y_data[idx]


# ==================================================
# 2. THE NEURAL NETWORK DEFINITION
# ==================================================
class KinematicToKineticLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2):
        super(KinematicToKineticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2) 

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_frame_output = lstm_out[:, -1, :] 
        prediction = self.fc(final_frame_output)
        return prediction


# ==================================================
# 3. THE TRAINING PIPELINE
# ==================================================
def train_model():
    print("Initializing Training Pipeline...")
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20
    
    dataset = BiomechanicsDataset(num_samples=1000, seq_length=60)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KinematicToKineticLSTM().to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    print(f"Training initialized on engine target: {device}\n" + "="*40)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        total_epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Mean Squared Error Loss: {total_epoch_loss:.5f}")
        
    model_filename = "gait_lstm.pt"
    torch.save(model.state_dict(), model_filename)
    print("="*40 + f"\nTraining Complete! Weights exported successfully to '{model_filename}'")

if __name__ == "__main__":
    train_model()