"""
Tennis Tactics Autoresearch: Fixed Data Pipeline
=================================================
Loads Grand Slam PBP data, creates features, provides fixed evaluation.

DO NOT MODIFY this file — it is the fixed evaluation and data infrastructure.
The agent modifies train_tennis.py only.

Usage:
    conda run -n ml python prepare_tennis.py         # verify data loading
    conda run -n ml python prepare_tennis.py --stats  # show data stats
"""

import os
import sys
import glob
import time
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300       # training time budget in seconds (5 minutes)
EVAL_BATCH_SIZE = 256   # batch size for evaluation
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PBP_DIR = os.path.join(BASE_DIR, "tennis_slam_pointbypoint")
MCP_DIR = os.path.join(BASE_DIR, "tennis_MatchChartingProject")
ATP_DIR = os.path.join(BASE_DIR, "tennis_atp")
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _get_year_from_filename(filepath):
    """Extract year from PBP filename like '2023-usopen-points.csv'."""
    basename = os.path.basename(filepath)
    try:
        return int(basename.split("-")[0])
    except (ValueError, IndexError):
        return None

def _list_singles_pbp_files():
    """Return sorted list of singles PBP point files (exclude doubles/mixed)."""
    all_files = sorted(glob.glob(os.path.join(PBP_DIR, "*-points.csv")))
    singles = [f for f in all_files
               if "doubles" not in os.path.basename(f)
               and "mixed" not in os.path.basename(f)]
    return singles

def load_pbp_data(verbose=True):
    """
    Load ALL Grand Slam PBP data and split by temporal year.
    
    Returns:
        dict with keys 'train', 'val', 'test' each containing a DataFrame.
        
    Split strategy (temporal, no data leakage):
        - Train: 2011–2022 (36 files, ~1.4M points)
        - Val:   2023      (4 files,  ~200K points)
        - Test:  2024      (2 files,  ~80K points)
    """
    files = _list_singles_pbp_files()
    assert len(files) > 0, f"No PBP files found in {PBP_DIR}. Clone the repo first."
    
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for f in files:
        year = _get_year_from_filename(f)
        if year is None:
            continue
        df = pd.read_csv(f, low_memory=False)
        if verbose:
            print(f"  {os.path.basename(f):45s} → {len(df):>7,} points  (year={year})")
        
        if year <= 2022:
            train_dfs.append(df)
        elif year == 2023:
            val_dfs.append(df)
        elif year >= 2024:
            test_dfs.append(df)
    
    splits = {}
    for name, dfs in [("train", train_dfs), ("val", val_dfs), ("test", test_dfs)]:
        if dfs:
            splits[name] = pd.concat(dfs, ignore_index=True)
        else:
            splits[name] = pd.DataFrame()
    
    if verbose:
        for name in ["train", "val", "test"]:
            print(f"  {name:5s}: {len(splits[name]):>9,} points")
    
    return splits

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

# These are the columns available in PBP data (65 columns):
#
# IDENTIFIERS:
#   match_id          - unique match identifier
#   PointNumber       - point number within the match
#   ElapsedTime       - time elapsed since match start
#
# MATCH STATE:
#   SetNo             - current set number
#   P1GamesWon        - games won by P1 in current set
#   P2GamesWon        - games won by P2 in current set
#   SetWinner         - who won the set (0 = ongoing)
#   GameNo            - current game number
#   GameWinner        - who won the game  (0 = ongoing)
#   P1Score           - P1 score in current game (0/15/30/40/AD)
#   P2Score           - P2 score in current game (0/15/30/40/AD)
#
# SERVE:
#   PointServer       - who is serving (1 or 2)
#   ServeNumber       - 1st or 2nd serve
#   Speed_KMH         - serve speed in km/h
#   Speed_MPH         - serve speed in mph
#   ServeIndicator    - 1=first serve, 2=second serve
#   Serve_Direction   - direction of serve
#   ServeWidth        - B=body, BC=body/center, BW=body/wide, C=center/T, W=wide
#   ServeDepth        - CTL=close to line, NCTL=not close to line
#   ServingTo         - deuce or ad court
#
# RETURN:
#   ReturnDepth       - D=deep, ND=not deep
#
# RALLY:
#   Rally             - rally description
#   RallyCount        - number of shots in the rally
#
# OUTCOMES:
#   PointWinner       - who won the point (1 or 2) ← THIS IS THE TARGET
#   P1Ace/P2Ace       - was it an ace
#   P1Winner/P2Winner - was it a winner
#   P1DoubleFault/P2DoubleFault - double fault
#   P1UnfErr/P2UnfErr - unforced error
#   P1ForcedError/P2ForcedError - forced error
#   Winner_FH/Winner_BH - forehand/backhand winner
#   WinnerType        - type of winner
#   WinnerShotType    - shot type of winner
#
# NET PLAY:
#   P1NetPoint/P2NetPoint     - was player at net
#   P1NetPointWon/P2NetPointWon - did they win the net point
#
# BREAK POINTS:
#   P1BreakPoint/P2BreakPoint       - is it a break point
#   P1BreakPointWon/P2BreakPointWon - was it converted
#   P1BreakPointMissed/P2BreakPointMissed
#
# SERVE STATS (cumulative):
#   P1FirstSrvIn/P2FirstSrvIn     - first serve in
#   P1FirstSrvWon/P2FirstSrvWon   - first serve won
#   P1SecondSrvIn/P2SecondSrvIn   - second serve in
#   P1SecondSrvWon/P2SecondSrvWon - second serve won
#
# MOMENTUM:
#   P1Momentum/P2Momentum - momentum metric
#   P1PointsWon/P2PointsWon - cumulative points won
#   P1TurningPoint/P2TurningPoint - match turning points
#
# DISTANCE:
#   P1DistanceRun/P2DistanceRun - distance run (meters)


def convert_score(s):
    """Convert game score string to numeric."""
    if pd.isna(s):
        return 0
    s_str = str(s).upper().strip()
    if s_str in ["0", "15", "30", "40"]:
        return int(s_str)
    if s_str == "AD":
        return 50
    return 0


def engineer_features(df):
    """
    Engineer features from raw PBP data.
    Returns feature matrix X (numpy), target vector y (numpy), feature names,
    and a metadata dict with tactical indicators (key_point, serve_width, etc.).
    
    This function is the core of the data pipeline. It converts raw PBP columns
    into numeric features suitable for model training.
    """
    # Filter valid points (must have a winner)
    df = df[df["PointWinner"].isin([1, 2])].copy()
    if len(df) == 0:
        return np.array([]), np.array([]), [], {}
    
    # Target: 0 if P1 won, 1 if P2 won
    y = (df["PointWinner"] - 1).values.astype(np.float32)
    
    # === Numeric features ===
    
    # Serve context
    df["PointServer_num"] = pd.to_numeric(df["PointServer"], errors="coerce").fillna(1).astype(float)
    df["ServeNumber_num"] = pd.to_numeric(df["ServeNumber"], errors="coerce").fillna(1).astype(float)
    
    # Score state
    df["P1Score_num"] = df["P1Score"].apply(convert_score).astype(float)
    df["P2Score_num"] = df["P2Score"].apply(convert_score).astype(float)
    df["ScoreDiff"] = df["P1Score_num"] - df["P2Score_num"]
    
    # Set/game state
    df["SetNo_num"] = pd.to_numeric(df["SetNo"], errors="coerce").fillna(1).astype(float)
    df["P1GamesWon_num"] = pd.to_numeric(df["P1GamesWon"], errors="coerce").fillna(0).astype(float)
    df["P2GamesWon_num"] = pd.to_numeric(df["P2GamesWon"], errors="coerce").fillna(0).astype(float)
    df["GameDiff"] = df["P1GamesWon_num"] - df["P2GamesWon_num"]
    
    # Momentum
    df["P1Momentum_num"] = pd.to_numeric(df["P1Momentum"], errors="coerce").fillna(0).astype(float)
    df["P2Momentum_num"] = pd.to_numeric(df["P2Momentum"], errors="coerce").fillna(0).astype(float)
    df["MomentumDiff"] = df["P1Momentum_num"] - df["P2Momentum_num"]
    
    # Points won (cumulative)
    df["P1PointsWon_num"] = pd.to_numeric(df["P1PointsWon"], errors="coerce").fillna(0).astype(float)
    df["P2PointsWon_num"] = pd.to_numeric(df["P2PointsWon"], errors="coerce").fillna(0).astype(float)
    df["PointsWonDiff"] = df["P1PointsWon_num"] - df["P2PointsWon_num"]
    
    # Rally context
    df["RallyCount_num"] = pd.to_numeric(df["RallyCount"], errors="coerce").fillna(0).astype(float)
    
    # Serve speed
    df["Speed_KMH_num"] = pd.to_numeric(df["Speed_KMH"], errors="coerce").fillna(0).astype(float)
    
    # Break point pressure
    df["P1BreakPoint_num"] = pd.to_numeric(df["P1BreakPoint"], errors="coerce").fillna(0).astype(float)
    df["P2BreakPoint_num"] = pd.to_numeric(df["P2BreakPoint"], errors="coerce").fillna(0).astype(float)
    df["IsBreakPoint"] = ((df["P1BreakPoint_num"] > 0) | (df["P2BreakPoint_num"] > 0)).astype(float)
    
    # Distance
    df["P1Distance_num"] = pd.to_numeric(df["P1DistanceRun"], errors="coerce").fillna(0).astype(float)
    df["P2Distance_num"] = pd.to_numeric(df["P2DistanceRun"], errors="coerce").fillna(0).astype(float)
    
    # Serve direction (categorical → numeric)
    le_dir = LabelEncoder()
    df["ServeDir_enc"] = le_dir.fit_transform(df["Serve_Direction"].astype(str))
    
    # Serve width (categorical → numeric)
    le_width = LabelEncoder()
    df["ServeWidth_enc"] = le_width.fit_transform(df["ServeWidth"].astype(str))
    
    # Serve depth (categorical → numeric)
    le_depth = LabelEncoder()
    df["ServeDepth_enc"] = le_depth.fit_transform(df["ServeDepth"].astype(str))
    
    # Return depth (categorical → numeric)
    le_ret = LabelEncoder()
    df["ReturnDepth_enc"] = le_ret.fit_transform(df["ReturnDepth"].astype(str))
    
    # === Derived features ===
    
    # Score pressure: higher when score is close and deep in the game
    df["ScorePressure"] = (df["P1Score_num"] + df["P2Score_num"]) / 100.0
    
    # Is tiebreak (approximation: both players at 6 games)
    df["IsTiebreak"] = ((df["P1GamesWon_num"] >= 6) & (df["P2GamesWon_num"] >= 6)).astype(float)
    
    # Server advantage: is the server ahead or behind
    df["ServerAhead"] = np.where(
        df["PointServer_num"] == 1,
        df["PointsWonDiff"],
        -df["PointsWonDiff"]
    )
    
    # === Key Point Indicator (for tactical analysis and weighted loss) ===
    # Key points: 30-30, 30-40, 40-30, 40-40 (Deuce), AD-40, 40-AD, 0-40, 40-0
    s1, s2 = df["P1Score_num"].values, df["P2Score_num"].values
    is_key_point = np.zeros(len(df), dtype=np.float32)
    # 30-30: next point creates break/game point
    is_key_point[(s1 == 30) & (s2 == 30)] = 1.0
    # 30-40 or 40-30: break point situations
    is_key_point[(s1 == 30) & (s2 == 40)] = 1.0
    is_key_point[(s1 == 40) & (s2 == 30)] = 1.0
    # Deuce and AD
    is_key_point[(s1 == 40) & (s2 == 40)] = 1.0
    is_key_point[(s1 == 50)] = 1.0  # AD-40
    is_key_point[(s2 == 50)] = 1.0  # 40-AD
    # 0-40 or 40-0 (triple break/game point)
    is_key_point[(s1 == 0) & (s2 == 40)] = 1.0
    is_key_point[(s1 == 40) & (s2 == 0)] = 1.0
    # Tiebreak points are all key
    is_key_point[df["IsTiebreak"].values == 1.0] = 1.0
    
    # Server won the point (for tactical analysis)
    server_won = np.where(
        df["PointServer_num"].values == 1,
        (y == 0).astype(np.float32),
        (y == 1).astype(np.float32),
    )
    
    # Feature list
    feature_cols = [
        "PointServer_num", "ServeNumber_num",
        "P1Score_num", "P2Score_num", "ScoreDiff",
        "SetNo_num", "P1GamesWon_num", "P2GamesWon_num", "GameDiff",
        "P1Momentum_num", "P2Momentum_num", "MomentumDiff",
        "P1PointsWon_num", "P2PointsWon_num", "PointsWonDiff",
        "RallyCount_num", "Speed_KMH_num",
        "P1BreakPoint_num", "P2BreakPoint_num", "IsBreakPoint",
        "P1Distance_num", "P2Distance_num",
        "ServeDir_enc", "ServeWidth_enc", "ServeDepth_enc", "ReturnDepth_enc",
        "ScorePressure", "IsTiebreak", "ServerAhead",
    ]
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Replace any remaining NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Metadata for tactical analysis (not part of features)
    metadata = {
        "is_key_point": is_key_point,
        "server_won": server_won,
        "serve_width_raw": df["ServeWidth"].values,
        "serve_number_raw": df["ServeNumber_num"].values,
        "rally_count_raw": df["RallyCount_num"].values,
        "p1_distance_raw": df["P1Distance_num"].values,
        "p2_distance_raw": df["P2Distance_num"].values,
    }
    
    return X, y, feature_cols, metadata


# ---------------------------------------------------------------------------
# Sequence Creation (for LSTM/Transformer models)
# ---------------------------------------------------------------------------

def create_sequences(X, y, seq_len=5):
    """
    Create overlapping sequences of length seq_len for sequential models.
    Each sequence predicts the outcome of the point after the sequence.
    
    Returns:
        X_seq: shape (N - seq_len, seq_len, num_features)
        y_seq: shape (N - seq_len,)
    """
    if len(X) <= seq_len:
        return np.array([]), np.array([])
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def make_dataloaders(X_train, y_train, X_val, y_val, batch_size=64, seq_len=None):
    """
    Create PyTorch DataLoaders from numpy arrays.
    
    If seq_len is provided, creates sequential data (for LSTM/Transformer).
    If seq_len is None, creates flat data (for MLP/boosting).
    
    Returns:
        train_loader, val_loader, input_dim
    """
    if seq_len is not None:
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
        train_X_t = torch.tensor(X_train_seq, dtype=torch.float32)
        train_y_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
        val_X_t = torch.tensor(X_val_seq, dtype=torch.float32)
        val_y_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
        input_dim = X_train.shape[1]
    else:
        train_X_t = torch.tensor(X_train, dtype=torch.float32)
        train_y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        val_X_t = torch.tensor(X_val, dtype=torch.float32)
        val_y_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        input_dim = X_train.shape[1]
    
    train_ds = TensorDataset(train_X_t, train_y_t)
    val_ds = TensorDataset(val_X_t, val_y_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, input_dim


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model, val_loader, device):
    """
    Fixed evaluation metric: Validation accuracy on point-winner prediction.
    Higher is better.
    
    The model must output logits or probabilities of shape (B, 1).
    Predictions > 0.5 are classified as class 1 (P2 wins).
    """
    model.eval()
    correct = 0
    total = 0
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_x)
        preds = (output > 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_auc(model, val_loader, device):
    """
    Secondary metric: AUC-ROC for more nuanced evaluation.
    """
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_probs = []
    all_labels = []
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        all_probs.extend(output.cpu().numpy().flatten())
        all_labels.extend(batch_y.numpy().flatten())
    try:
        return roc_auc_score(all_labels, all_probs)
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Convenience: Full Pipeline
# ---------------------------------------------------------------------------

def prepare_data(verbose=True, max_train_points=None, scale=True):
    """
    Full data preparation pipeline.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler,
        meta_train, meta_val, meta_test
    
    meta_* dicts contain tactical indicators:
        - is_key_point: binary array (1.0 for 30-30, 30-40, deuce, tiebreak, etc.)
        - server_won: binary array (1.0 if server won the point)
        - serve_width_raw: string array of serve width codes (W/C/B/BC/BW)
        - serve_number_raw: 1 or 2 (first/second serve)
        - rally_count_raw: number of shots
        - p1_distance_raw / p2_distance_raw: meters run
    """
    if verbose:
        print("Loading Grand Slam PBP data...")
    splits = load_pbp_data(verbose=verbose)
    
    if verbose:
        print("\nEngineering features...")
    
    X_train, y_train, feature_names, meta_train = engineer_features(splits["train"])
    X_val, y_val, _, meta_val = engineer_features(splits["val"])
    X_test, y_test, _, meta_test = engineer_features(splits["test"])
    
    if verbose:
        print(f"  Train features: {X_train.shape}")
        print(f"  Val features:   {X_val.shape}")
        print(f"  Test features:  {X_test.shape}")
        print(f"  Feature count:  {len(feature_names)}")
        print(f"  Features: {feature_names}")
        # Key point stats
        kp_train = meta_train["is_key_point"].sum()
        print(f"  Train key points: {int(kp_train):,} ({kp_train/len(y_train):.1%})")
    
    # Optionally limit training data for faster iteration
    if max_train_points is not None and len(X_train) > max_train_points:
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(len(X_train), max_train_points, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        for k in meta_train:
            meta_train[k] = meta_train[k][idx]
        if verbose:
            print(f"  Subsampled train to {max_train_points:,} points")
    
    # Scale features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        # Convert back to float32
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler, meta_train, meta_val, meta_test


# ---------------------------------------------------------------------------
# Main (for verification / stats)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis Autoresearch: Data Pipeline")
    parser.add_argument("--stats", action="store_true", help="Show detailed data statistics")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TENNIS AUTORESEARCH: DATA PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Verify environment
    print(f"\nPyTorch: {torch.__version__}")
    device_type = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_type}")
    print(f"Time budget: {TIME_BUDGET}s")
    
    # Load and process data
    print()
    X_train, y_train, X_val, y_val, X_test, y_test, features, scaler, \
        meta_train, meta_val, meta_test = prepare_data(verbose=True)
    
    if args.stats:
        print("\n" + "=" * 60)
        print("DATA STATISTICS")
        print("=" * 60)
        print(f"\nTarget distribution:")
        for name, y_data in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            if len(y_data) > 0:
                p1_wins = (y_data == 0).sum()
                p2_wins = (y_data == 1).sum()
                print(f"  {name}: P1 wins={p1_wins:,} ({p1_wins/len(y_data):.1%}), "
                      f"P2 wins={p2_wins:,} ({p2_wins/len(y_data):.1%})")
        
        print(f"\nFeature statistics (train, scaled):")
        for i, name in enumerate(features):
            col = X_train[:, i]
            print(f"  {name:25s}: mean={col.mean():>8.3f}  std={col.std():>8.3f}  "
                  f"min={col.min():>8.3f}  max={col.max():>8.3f}")
    
    print("\n✅ Data pipeline verified! Ready for training.")
    print(f"   Run: conda run -n ml python train_tennis.py")
