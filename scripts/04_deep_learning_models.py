"""
Phase 4: Matchup & Deep Learning Models
========================================
Extracts Grand Slam PBP data to build predictive models for point outcomes.
Implements XGBoost, LightGBM, LSTMs for momentum modeling, Transformers for
rally/sequence prediction, and Autoencoders for tactical anomaly detection.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Paths & Setup ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_DIR = os.path.join(BASE_DIR, "tennis_slam_pointbypoint")
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")

print("=" * 60)
print("PHASE 4: DEEP LEARNING & PREDICTIVE MODELING")
print("=" * 60)

# Build a small sample dataset from recent Grand Slams to train efficiently
print("\n[1/6] Loading Grand Slam Point-by-Point Data...")
files = [
    os.path.join(PBP_DIR, "2023-usopen-points.csv"),
    os.path.join(PBP_DIR, "2024-ausopen-points.csv")
]

dfs = []
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

df_pbp = pd.concat(dfs, ignore_index=True)
print(f"  Loaded {len(df_pbp):,} points from {len(dfs)} tournaments.")

# ── 2. Data Preprocessing ───────────────────────────────────────────────────
print("\n[2/6] Feature Engineering & Preprocessing...")

# We want to predict PointWinner (1 or 2)
# Filter valid points
df_pbp = df_pbp[df_pbp['PointWinner'].isin([1, 2])].copy()
df_pbp['target'] = df_pbp['PointWinner'] - 1  # 0 or 1

# Score mapping
def convert_score(s):
    if pd.isna(s): return 0
    s_str = str(s).upper()
    if s_str in ["0", "15", "30", "40"]: return int(s_str)
    if s_str == "AD": return 50
    return 0

df_pbp["P1Score_num"] = df_pbp["P1Score"].apply(convert_score)
df_pbp["P2Score_num"] = df_pbp["P2Score"].apply(convert_score)

# Fill momentum and basic features
df_pbp["P1Momentum"] = pd.to_numeric(df_pbp["P1Momentum"], errors="coerce").fillna(0)
df_pbp["P2Momentum"] = pd.to_numeric(df_pbp["P2Momentum"], errors="coerce").fillna(0)
df_pbp["ServeNumber"] = pd.to_numeric(df_pbp["ServeNumber"], errors="coerce").fillna(1)
df_pbp["PointServer"] = pd.to_numeric(df_pbp["PointServer"], errors="coerce").fillna(1)
df_pbp["RallyCount"] = pd.to_numeric(df_pbp["RallyCount"], errors="coerce").fillna(0)

# Categorical mapping for Serve Direction
le_dir = LabelEncoder()
df_pbp["Serve_Direction_Enc"] = le_dir.fit_transform(df_pbp["Serve_Direction"].astype(str))

features = [
    "PointServer", "ServeNumber", "P1Score_num", "P2Score_num", 
    "P1Momentum", "P2Momentum", "Serve_Direction_Enc", "RallyCount"
]

X = df_pbp[features].values
y = df_pbp["target"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"  Train set: {X_train.shape[0]:,} points | Test set: {X_test.shape[0]:,} points")

# ── 3. Gradient Boosting Models ─────────────────────────────────────────────
print("\n[3/6] Training HistGradientBoosting (scikit-learn) Baseline Model...")

# HistGradientBoostingClassifier (Alternative to LightGBM/XGBoost that doesn't need external libomp)
hgb_model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, random_state=42)
hgb_model.fit(X_train, y_train)
hgb_preds = hgb_model.predict(X_test)
hgb_acc = accuracy_score(y_test, hgb_preds)
print(f"  ✓ HistGradientBoosting Accuracy: {hgb_acc:.3%}")

# ── 4. Sequence Modeling: LSTM & Transformer ─────────────────────────────────
print("\n[4/6] Deep Learning: LSTMs & Transformers for Momentum Modeling...")

# Prepare sequences for PyTorch (looking back N points to predict the N+1th point)
SEQ_LEN = 5
def create_sequences(data, labels, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(labels[i+seq_len])
    return np.array(xs), np.array(ys)

# Subsample for speed during this demonstration
sample_limit = 25000 
X_seq_data = X_scaled[:sample_limit]
y_seq_data = y[:sample_limit]

X_seq, y_seq = create_sequences(X_seq_data, y_seq_data, SEQ_LEN)
X_seq_t = torch.tensor(X_seq, dtype=torch.float32)
y_seq_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_seq_t, y_seq_t)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# A. LSTM Model
class MomentumLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

lstm_model = MomentumLSTM(input_dim=len(features), hidden_dim=32)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

print("  Training LSTM (Momentum Tracker)...")
lstm_model.train()
for epoch in range(3):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = lstm_model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

lstm_model.eval()
correct = 0
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = (lstm_model(batch_x) > 0.5).float()
        correct += (preds == batch_y).sum().item()
lstm_acc = correct / val_size
print(f"  ✓ LSTM Validation Accuracy: {lstm_acc:.3%}")

# B. Transformer Model
class RallyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        # Pool sequences
        out = out.mean(dim=1)
        out = self.fc(out)
        return self.sigmoid(out)

tf_model = RallyTransformer(input_dim=len(features))
optimizer_tf = torch.optim.Adam(tf_model.parameters(), lr=0.005)

print("  Training Transformer (Rally Attention)...")
tf_model.train()
for epoch in range(3):
    for batch_x, batch_y in train_loader:
        optimizer_tf.zero_grad()
        out = tf_model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer_tf.step()

tf_model.eval()
correct_tf = 0
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = (tf_model(batch_x) > 0.5).float()
        correct_tf += (preds == batch_y).sum().item()
tf_acc = correct_tf / val_size
print(f"  ✓ Transformer Validation Accuracy: {tf_acc:.3%}")

# ── 5. Autoencoder for Tactical Anomalies ───────────────────────────────────
print("\n[5/6] Autoencoder: Tactical Anomaly Detection for Top Players...")

# Filter points where Player 1 won to learn the "winning tactical baseline"
winning_pts = df_pbp[df_pbp["target"] == 0][features].values
winning_scaled = scaler.transform(winning_pts)
winning_t = torch.tensor(winning_scaled[:10000], dtype=torch.float32)

class TacticalAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

ae_model = TacticalAutoencoder(input_dim=len(features))
optimizer_ae = torch.optim.Adam(ae_model.parameters(), lr=0.01)
criterion_ae = nn.MSELoss()

print("  Training Autoencoder on winning points baseline...")
ae_model.train()
for epoch in range(5):
    optimizer_ae.zero_grad()
    reconstructed = ae_model(winning_t)
    loss = criterion_ae(reconstructed, winning_t)
    loss.backward()
    optimizer_ae.step()

# Compute reconstruction error to find anomalies (tactical deviations)
ae_model.eval()
with torch.no_grad():
    sample_reconstructed = ae_model(winning_t)
    mse = torch.mean((winning_t - sample_reconstructed)**2, dim=1).numpy()

threshold = np.percentile(mse, 95)
anomalies = np.sum(mse > threshold)
print(f"  ✓ Autoencoder learned baseline. Detected {anomalies} anomalous points (tactical deviations) out of {len(mse)}.")

# ── 6. Save Model Metrics & Metadata ────────────────────────────────────────
print("\n[6/6] Generating DL Scouting Report Metadata...")

dl_report = {
    "hist_gb_accuracy": float(hgb_acc),
    "lstm_momentum_accuracy": float(lstm_acc),
    "transformer_attention_accuracy": float(tf_acc),
    "autoencoder_anomaly_threshold": float(threshold)
}

with open(os.path.join(DATA_DIR, "deep_learning_metrics.json"), "w") as f:
    json.dump(dl_report, f, indent=2)

print(f"  ✓ deep_learning_metrics.json saved!")

print("\n✅ Phase 4 complete!")
