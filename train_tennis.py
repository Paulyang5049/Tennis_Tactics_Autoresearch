"""
Tennis Tactics Autoresearch: Training Script
=============================================
This is the file the AI agent modifies. Everything is fair game:
- Model architecture (LSTM depth, Transformer heads, hybrid, etc.)
- Hyperparameters (LR, batch size, epochs, regularization)
- Feature selection and engineering
- Sequence length for temporal models
- Key-point weighting in loss function
- Training strategy (curriculum, augmentation, etc.)

Usage:
    conda run -n ml python train_tennis.py

The output reports val_accuracy and tactical insights (higher accuracy is better).

Sport Analytics Goals:
- Learn serve direction effectiveness on key points (30-30, 30-40, deuce)
- Capture serve+1 tactical patterns
- Model rally length and distance as physical endurance indicators
- Weight key points higher — they separate champions from average players
"""

import os
import gc
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prepare_tennis import (
    TIME_BUDGET, RANDOM_SEED,
    prepare_data, make_dataloaders, create_sequences,
    evaluate_accuracy, evaluate_auc,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these)
# ---------------------------------------------------------------------------

# Data
MAX_TRAIN_POINTS = None     # None = use all data, or set to e.g. 200000 for faster iteration
SEQ_LEN = 10                # lookback window for sequential models (LSTM/Transformer)
BATCH_SIZE = 128             # training batch size

# Architecture
HIDDEN_DIM = 64             # hidden dimension for LSTM/Transformer
NUM_LAYERS = 2              # number of layers
D_MODEL = 64                # Transformer model dimension
NHEAD = 4                   # Transformer attention heads
DROPOUT = 0.1               # dropout rate

# Optimization
LEARNING_RATE = 0.001       # initial learning rate
WEIGHT_DECAY = 1e-5         # L2 regularization

# Key-point weighting (Sport Analytics Principle #1)
# Key points (30-30, 30-40, deuce, tiebreak) are weighted more in the loss
# because this is where tactical decisions matter most
KEY_POINT_WEIGHT = 2.0      # weight for key points vs 1.0 for regular points

# Multi-task learning (Sport Analytics Principle #2)
# Add a secondary task: predicting the optimal serve direction
SERVE_DIR_WEIGHT = 0.5      # weight for the auxiliary serve direction loss

# Model selection: "lstm", "transformer", "hybrid", or "mtl_lstm"
MODEL_TYPE = "mtl_lstm"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MomentumLSTM(nn.Module):
    """LSTM for sequential point prediction with momentum tracking."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        lstm_out, (hn, _) = self.lstm(x)
        # Use last hidden state
        out = self.norm(hn[-1])
        return self.fc(out)


class RallyTransformer(nn.Module):
    """Transformer for rally sequence prediction."""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        B, T, _ = x.shape
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :T, :]
        out = self.transformer(x)
        # Mean pooling over sequence
        out = out.mean(dim=1)
        return self.fc(out)


class HybridModel(nn.Module):
    """LSTM + Attention hybrid for momentum + rally modeling."""
    def __init__(self, input_dim, hidden_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        tf_out = self.transformer(lstm_out)
        out = tf_out.mean(dim=1)
        return self.fc(out)


class MTLLSTM(nn.Module):
    """Multi-Task LSTM predicting point winner AND serve direction."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1, num_serve_dirs=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Primary task: Point Winner (binary)
        self.fc_winner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Auxiliary task: Serve Direction (multi-class)
        self.fc_serve_dir = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_serve_dirs),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        lstm_out, (hn, _) = self.lstm(x)
        out = self.norm(hn[-1])
        winner_pred = self.fc_winner(out)
        serve_dir_pred = self.fc_serve_dir(out)
        return winner_pred, serve_dir_pred


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Detect device
device_type = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"Device: {device_type}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Model type: {MODEL_TYPE}")
print(f"Key-point weight: {KEY_POINT_WEIGHT}x")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

print("\n--- Data Loading ---")
X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler, \
    meta_train, meta_val, meta_test = prepare_data(
    verbose=True, max_train_points=MAX_TRAIN_POINTS
)

input_dim = X_train.shape[1]
print(f"\nInput dimension: {input_dim}")
print(f"Features: {feature_names}")

# Create sequential data for LSTM/Transformer, including key-point weights
from prepare_tennis import create_sequences

# Build sequence data
X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQ_LEN)

# Build key-point weight sequences (aligned with targets)
_, kp_train_seq = create_sequences(
    meta_train["is_key_point"].reshape(-1, 1),
    meta_train["is_key_point"],
    SEQ_LEN
)
_, kp_val_seq = create_sequences(
    meta_val["is_key_point"].reshape(-1, 1),
    meta_val["is_key_point"],
    SEQ_LEN
)

# Build serve direction sequences
serve_width_map = {"B": 0, "BC": 1, "BW": 2, "C": 3, "W": 4}
def encode_serve_width(arr):
    return np.array([serve_width_map.get(str(x).strip(), 3) for x in arr])

sd_train = encode_serve_width(meta_train["serve_width_raw"])
sd_val = encode_serve_width(meta_val["serve_width_raw"])

_, sd_train_seq = create_sequences(sd_train.reshape(-1, 1), sd_train, SEQ_LEN)
_, sd_val_seq = create_sequences(sd_val.reshape(-1, 1), sd_val, SEQ_LEN)

# Convert to tensors
train_X_t = torch.tensor(X_train_seq, dtype=torch.float32)
train_y_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
train_kp_t = torch.tensor(kp_train_seq, dtype=torch.float32).unsqueeze(1)
train_sd_t = torch.tensor(sd_train_seq, dtype=torch.long)

val_X_t = torch.tensor(X_val_seq, dtype=torch.float32)
val_y_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
val_kp_t = torch.tensor(kp_val_seq, dtype=torch.float32).unsqueeze(1)
val_sd_t = torch.tensor(sd_val_seq, dtype=torch.long)

# Create dataloaders WITH key-point weights and serve dirs
train_ds = TensorDataset(train_X_t, train_y_t, train_kp_t, train_sd_t)
val_ds = TensorDataset(val_X_t, val_y_t, val_kp_t, val_sd_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader_kp = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Standard val_loader for fixed evaluation metric (no key-point weights)
val_ds_std = TensorDataset(val_X_t, val_y_t)
val_loader = DataLoader(val_ds_std, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Key pts in train seqs: {int(train_kp_t.sum()):,} / {len(train_kp_t):,} "
      f"({train_kp_t.mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print("\n--- Model ---")
if MODEL_TYPE == "lstm":
    model = MomentumLSTM(input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)
elif MODEL_TYPE == "transformer":
    model = RallyTransformer(input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT)
elif MODEL_TYPE == "hybrid":
    model = HybridModel(input_dim, hidden_dim=HIDDEN_DIM, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT)
elif MODEL_TYPE == "mtl_lstm":
    model = MTLLSTM(input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, num_serve_dirs=5)
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model = model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# OneCycleLR for better convergence
estimated_steps = max(len(train_loader) * 3, 1000)  # ~3 epochs estimate
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=estimated_steps * 2,  # overestimate to avoid index error
    pct_start=0.1,
    anneal_strategy="cos",
)

# ---------------------------------------------------------------------------
# Key-Point Weighted Loss (Sport Analytics Principle #1)
# ---------------------------------------------------------------------------

def weighted_bce_loss(output, target, is_key_point):
    """
    BCE loss with higher weight on key points (30-30, 30-40, deuce, tiebreak).
    
    Rationale: Key points are where tactical decisions matter most.
    A model that's 2% better on key points is more valuable than one
    that's 2% better on 0-0 points.
    """
    weight = torch.where(is_key_point > 0.5, KEY_POINT_WEIGHT, 1.0)
    loss = F.binary_cross_entropy(output, target, weight=weight)
    return loss

serve_dir_criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Training Loop (time-budgeted)
# ---------------------------------------------------------------------------

# Helper to handle multi-task output for evaluation
def get_winner_preds(model, dataloader, device):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

print("\n--- Training ---")
t_training_start = time.time()
total_training_time = 0.0
step = 0
epoch = 0
best_val_acc = 0.0
smooth_loss = 0.0

# GC management
gc.collect()
gc.freeze()
gc.disable()

model.train()
while True:
    epoch += 1
    epoch_loss = 0.0
    epoch_steps = 0
    
    for batch_x, batch_y, batch_kp, batch_sd in train_loader:
        t_step_start = time.time()
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_kp = batch_kp.to(device)
        batch_sd = batch_sd.to(device)
        
        optimizer.zero_grad()
        
        if MODEL_TYPE.startswith("mtl"):
            output_winner, output_serve_dir = model(batch_x)
            loss_winner = weighted_bce_loss(output_winner, batch_y, batch_kp)
            loss_serve = serve_dir_criterion(output_serve_dir, batch_sd)
            loss = loss_winner + SERVE_DIR_WEIGHT * loss_serve
        else:
            output = model(batch_x)
            loss = weighted_bce_loss(output, batch_y, batch_kp)
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Sync device
        if device_type == "mps":
            torch.mps.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        
        t_step_end = time.time()
        dt = t_step_end - t_step_start
        
        # Don't count warmup steps
        if step > 5:
            total_training_time += dt
        
        train_loss_f = loss.item()
        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss_f
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        
        step += 1
        epoch_steps += 1
        epoch_loss += train_loss_f
        
        # Fast fail
        if train_loss_f > 10:
            print("FAIL: Loss exploded")
            sys.exit(1)
        
        # Log every 100 steps
        if step % 100 == 0:
            pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
            remaining = max(0, TIME_BUDGET - total_training_time)
            lr = optimizer.param_groups[0]["lr"]
            print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.6f} | "
                  f"lr: {lr:.6f} | dt: {dt*1000:.0f}ms | "
                  f"epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)
        
        # Time check
        if step > 5 and total_training_time >= TIME_BUDGET:
            break
    
    # End of epoch eval
    if step > 5 and total_training_time < TIME_BUDGET:
        model.eval()
        preds, targets = get_winner_preds(model, val_loader, device)
        val_acc = (preds.round() == targets).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"\n  Epoch {epoch} done | avg_loss: {epoch_loss/max(epoch_steps,1):.6f} | "
              f"val_acc: {val_acc:.4f} | best: {best_val_acc:.4f}")
        model.train()
    
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final Evaluation
# ---------------------------------------------------------------------------

print("\n--- Final Evaluation ---")
model.eval()

preds, targets = get_winner_preds(model, val_loader, device)
val_accuracy = (preds.round() == targets).mean()

from sklearn.metrics import roc_auc_score
try:
    val_auc = roc_auc_score(targets, preds)
except:
    val_auc = 0.5

# Key-point specific accuracy (Sport Analytics Principle #1)
with torch.no_grad():
    kp_correct = 0
    kp_total = 0
    reg_correct = 0
    reg_total = 0
    for batch_x, batch_y, batch_kp, batch_sd in val_loader_kp:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_kp = batch_kp.to(device)
        output = model(batch_x)
        if isinstance(output, tuple):
            output = output[0]
        preds = (output > 0.5).float()
        correct = (preds == batch_y).float()
        
        kp_mask = batch_kp.squeeze(1) > 0.5
        reg_mask = ~kp_mask
        
        kp_correct += (correct.squeeze() * kp_mask.float()).sum().item()
        kp_total += kp_mask.sum().item()
        reg_correct += (correct.squeeze() * reg_mask.float()).sum().item()
        reg_total += reg_mask.sum().item()

kp_accuracy = kp_correct / kp_total if kp_total > 0 else 0.0
reg_accuracy = reg_correct / reg_total if reg_total > 0 else 0.0

t_end = time.time()

# Memory reporting
if device_type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

# ---------------------------------------------------------------------------
# Tactical Analysis Summary
# ---------------------------------------------------------------------------

print("\n--- Tactical Analysis ---")
print(f"Overall val accuracy:      {val_accuracy:.4f}")
print(f"Key-point accuracy:        {kp_accuracy:.4f} ({int(kp_total):,} pts)")
print(f"Regular-point accuracy:    {reg_accuracy:.4f} ({int(reg_total):,} pts)")
print(f"Key vs Regular gap:        {(kp_accuracy - reg_accuracy)*100:+.2f}%")

# Serve direction effectiveness on val set (from metadata)
print("\n--- Serve Width → Server Win Rate (Val Set, Empirical) ---")
sw_raw = meta_val["serve_width_raw"]
svr_won = meta_val["server_won"]
for width in ["W", "C", "B", "BC", "BW"]:
    mask = sw_raw == width
    if mask.sum() > 50:
        win_rate = svr_won[mask].mean()
        kp_mask = mask & (meta_val["is_key_point"] > 0.5)
        kp_win = svr_won[kp_mask].mean() if kp_mask.sum() > 20 else float("nan")
        print(f"  {width:3s}: overall {win_rate:.1%} | key-pts {kp_win:.1%} ({int(kp_mask.sum())} pts)")

# Rally length vs win rate
print("\n--- Rally Length → Server Win Rate (Val Set) ---")
rc = meta_val["rally_count_raw"]
for label, lo, hi in [("Short (1-3)", 1, 3), ("Medium (4-6)", 4, 6), ("Long (7+)", 7, 999)]:
    mask = (rc >= lo) & (rc <= hi)
    if mask.sum() > 50:
        win_rate = svr_won[mask].mean()
        print(f"  {label:15s}: server wins {win_rate:.1%} ({int(mask.sum()):,} pts)")

# Distance indicators
print("\n--- Distance Run (Val Set) ---")
d1 = meta_val["p1_distance_raw"]
d2 = meta_val["p2_distance_raw"]
dmask = (d1 > 0) & (d2 > 0)
if dmask.sum() > 100:
    ratio = d1[dmask] / (d2[dmask] + 1e-6)
    print(f"  Mean P1 distance: {d1[dmask].mean():.1f}m")
    print(f"  Mean P2 distance: {d2[dmask].mean():.1f}m")
    # Correlation: does running more correlate with losing?
    p1_more = (d1[dmask] > d2[dmask])
    # If P1 runs more, does P1 win less?
    y_val_flat = meta_val["server_won"]  # just check P1 win rate
    # Actually let's do point winner
    p1_won_when_ran_more = (1 - meta_val["server_won"][dmask][p1_more]).mean()  # rough proxy
    print(f"  Player who runs more → wins {p1_won_when_ran_more:.1%} of points (lower = disadvantage)")

# ---------------------------------------------------------------------------
# Final Output (for autoresearch agent to parse)
# ---------------------------------------------------------------------------

print("\n---")
print(f"val_accuracy:     {val_accuracy:.6f}")
print(f"val_auc:          {val_auc:.6f}")
print(f"kp_accuracy:      {kp_accuracy:.6f}")
print(f"reg_accuracy:     {reg_accuracy:.6f}")
print(f"kp_gap:           {(kp_accuracy - reg_accuracy)*100:+.2f}%")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_steps:      {step}")
print(f"num_params:       {num_params}")
print(f"model_type:       {MODEL_TYPE}")
print(f"hidden_dim:       {HIDDEN_DIM}")
print(f"num_layers:       {NUM_LAYERS}")
print(f"seq_len:          {SEQ_LEN}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"learning_rate:    {LEARNING_RATE}")
print(f"key_point_weight: {KEY_POINT_WEIGHT}")
print(f"epochs:           {epoch}")
