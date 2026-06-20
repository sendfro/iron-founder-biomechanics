# main.py (Dual-Brain Cloud Inference Node)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List
import os

app = FastAPI(title="Iron Founder: Hybrid Biomechanics API")

# ==================================================
# 1. PAYLOAD DEFINITIONS
# ==================================================
class FrameData(BaseModel):
    frame: int
    keypoints: List[List[float]]
    confidences: List[float]

class TrajectoryPayload(BaseModel):
    trajectory: List[FrameData]

class ImagePayload(BaseModel):
    # Streamlit will send a flattened 64x64x3 image array (12,288 pixel values)
    image_pixels: List[float] 

# ==================================================
# 2. NEURAL NETWORK ARCHITECTURES
# ==================================================
# --- Brain 1: The Kinetic LSTM ---
class KinematicToKineticLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2):
        super(KinematicToKineticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2) 

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# --- Brain 2: The Structural Autoencoder ---
class KneeAnomalyAutoencoder(nn.Module):
    def __init__(self):
        super(KneeAnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==================================================
# 3. INITIALIZE MODELS & LOAD WEIGHTS
# ==================================================
# Load LSTM
lstm_model = KinematicToKineticLSTM()
if os.path.exists("gait_lstm.pt"):
    lstm_model.load_state_dict(torch.load("gait_lstm.pt", map_location=torch.device('cpu')))
    print("✅ LSTM Weights Loaded Successfully")
lstm_model.eval()

# Load Autoencoder
autoencoder_model = KneeAnomalyAutoencoder()
if os.path.exists("knee_autoencoder.pt"):
    autoencoder_model.load_state_dict(torch.load("knee_autoencoder.pt", map_location=torch.device('cpu')))
    print("✅ Autoencoder Weights Loaded Successfully")
autoencoder_model.eval()

# ==================================================
# 4. API ENDPOINTS
# ==================================================
@app.post("/predict_kinetics")
async def predict_kinetics(payload: TrajectoryPayload):
    if not payload.trajectory: raise HTTPException(status_code=400, detail="Empty trajectory")
    try:
        sequence = []
        for frame_data in payload.trajectory:
            flat_kps = [coord for kp in frame_data.keypoints for coord in kp]
            if len(flat_kps) == 34: sequence.append(flat_kps)
        
        if not sequence: raise HTTPException(status_code=400, detail="No valid sequences")

        input_tensor = torch.tensor([sequence], dtype=torch.float32)
        with torch.no_grad():
            prediction = lstm_model(input_tensor)
        
        raw_grf = float(prediction[0][0].item())
        raw_lateral = float(prediction[0][1].item())
        
        lateral_load = round(min(2.5, max(0.0, abs(raw_lateral))), 2)
        return {
            "status": "success",
            "generative_physics": {
                "peak_grf_bodyweight": round(abs(raw_grf) + 1.0, 2),
                "lateral_knee_load_multiplier": lateral_load,
                "predicted_acl_strain": "High" if lateral_load > 0.8 else "Nominal"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_anomaly")
async def detect_anomaly(payload: ImagePayload):
    try:
        # 1. Convert the incoming list of 12,288 flat pixels back into a PyTorch Tensor
        pixel_data = torch.tensor(payload.image_pixels, dtype=torch.float32)
        if pixel_data.shape[0] != 12288:
            raise ValueError(f"Expected 12288 pixels, got {pixel_data.shape[0]}")
        
        # 2. Reshape into an image: (Batch Size 1, 3 Color Channels, 64 Height, 64 Width)
        image_tensor = pixel_data.view(1, 3, 64, 64)

        # 3. Pass through the Autoencoder to see if it can successfully redraw the knee
        with torch.no_grad():
            reconstructed = autoencoder_model(image_tensor)
            
        # 4. Calculate the Reconstruction Error (Mean Squared Error)
        mse_loss = nn.MSELoss()(reconstructed, image_tensor).item()
        
        # 5. If the error is high, the anatomy is structurally anomalous
        threshold = 0.08000 # Our learned threshold
        is_anomaly = mse_loss > threshold
        
        return {
            "status": "success",
            "structural_analysis": {
                "reconstruction_error": round(mse_loss, 5),
                "anomaly_detected": is_anomaly,
                "tissue_health": "Structural Anomaly Detected" if is_anomaly else "Nominal"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))