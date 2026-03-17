"""
Phase 1: Ingest & Clean
========================
Merges 20 years of ATP match data (2005-2024), joins player bios,
computes derived performance metrics, and exports a clean master CSV.

Usage:
    source .venv/bin/activate
    python scripts/01_ingest_and_clean.py
"""

import os
import glob
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATP_DIR = os.path.join(BASE_DIR, "tennis_atp")
OUTPUT_DIR = os.path.join(BASE_DIR, "dashboard_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Merge annual match files (2005-2024) ──────────────────────────────────
print("=" * 60)
print("PHASE 1: INGEST & CLEAN")
print("=" * 60)

print("\n[1/6] Merging ATP tour-level match files (2005-2024)...")
match_files = sorted(glob.glob(os.path.join(ATP_DIR, "atp_matches_20[0-2][0-9].csv")))
match_files = [f for f in match_files if 2005 <= int(os.path.basename(f).split("_")[2].split(".")[0]) <= 2024]

dfs = []
for f in match_files:
    df = pd.read_csv(f, low_memory=False)
    year = int(os.path.basename(f).split("_")[2].split(".")[0])
    df["year"] = year
    dfs.append(df)
    print(f"  ✓ {os.path.basename(f):40s} → {len(df):>5,} matches")

df_matches = pd.concat(dfs, ignore_index=True)
print(f"\n  Total matches loaded: {len(df_matches):,}")

# ── 2. Load player bios ─────────────────────────────────────────────────────
print("\n[2/6] Loading player bios...")
df_players = pd.read_csv(os.path.join(ATP_DIR, "atp_players.csv"), low_memory=False)
df_players.rename(columns={"player_id": "player_id"}, inplace=True)
print(f"  Total players in bio file: {len(df_players):,}")

# ── 3. Clean data ───────────────────────────────────────────────────────────
print("\n[3/6] Cleaning data...")

# 3a. Drop matches with no serve stats (pre-1991 Davis Cup, etc.)
before = len(df_matches)
df_matches = df_matches.dropna(subset=["w_svpt", "l_svpt"])
print(f"  Dropped {before - len(df_matches):,} matches with missing serve stats")

# 3b. Drop obvious outliers
df_matches = df_matches[
    (df_matches["minutes"].isna()) | (df_matches["minutes"] >= 20)
]
print(f"  After outlier removal: {len(df_matches):,} matches remain")

# 3c. Extract surface from tourney info
df_matches["surface"] = df_matches["surface"].fillna("Unknown")

# ── 4. Derive performance metrics ───────────────────────────────────────────
print("\n[4/6] Computing derived performance metrics...")

def safe_div(a, b):
    """Division returning NaN when denominator is zero."""
    return np.where(b > 0, a / b, np.nan)

# Winner metrics
df_matches["w_1st_serve_pct"]     = safe_div(df_matches["w_1stIn"], df_matches["w_svpt"])
df_matches["w_1st_serve_win_pct"] = safe_div(df_matches["w_1stWon"], df_matches["w_1stIn"])
df_matches["w_2nd_serve_win_pct"] = safe_div(df_matches["w_2ndWon"], df_matches["w_svpt"] - df_matches["w_1stIn"])
df_matches["w_ace_rate"]          = safe_div(df_matches["w_ace"], df_matches["w_svpt"])
df_matches["w_df_rate"]           = safe_div(df_matches["w_df"], df_matches["w_svpt"])
df_matches["w_bp_save_pct"]       = safe_div(df_matches["w_bpSaved"], df_matches["w_bpFaced"])
df_matches["w_svc_pts_won"]       = safe_div(df_matches["w_1stWon"] + df_matches["w_2ndWon"], df_matches["w_svpt"])

# Loser metrics
df_matches["l_1st_serve_pct"]     = safe_div(df_matches["l_1stIn"], df_matches["l_svpt"])
df_matches["l_1st_serve_win_pct"] = safe_div(df_matches["l_1stWon"], df_matches["l_1stIn"])
df_matches["l_2nd_serve_win_pct"] = safe_div(df_matches["l_2ndWon"], df_matches["l_svpt"] - df_matches["l_1stIn"])
df_matches["l_ace_rate"]          = safe_div(df_matches["l_ace"], df_matches["l_svpt"])
df_matches["l_df_rate"]           = safe_div(df_matches["l_df"], df_matches["l_svpt"])
df_matches["l_bp_save_pct"]       = safe_div(df_matches["l_bpSaved"], df_matches["l_bpFaced"])
df_matches["l_svc_pts_won"]       = safe_div(df_matches["l_1stWon"] + df_matches["l_2ndWon"], df_matches["l_svpt"])

# Return points won (derived from opponent serve stats)
df_matches["w_return_pts_won_pct"] = safe_div(
    df_matches["l_svpt"] - df_matches["l_1stWon"] - df_matches["l_2ndWon"],
    df_matches["l_svpt"]
)
df_matches["l_return_pts_won_pct"] = safe_div(
    df_matches["w_svpt"] - df_matches["w_1stWon"] - df_matches["w_2ndWon"],
    df_matches["w_svpt"]
)

# Dominance ratio
df_matches["w_dominance_ratio"] = safe_div(df_matches["w_svc_pts_won"], df_matches["w_return_pts_won_pct"])
df_matches["l_dominance_ratio"] = safe_div(df_matches["l_svc_pts_won"], df_matches["l_return_pts_won_pct"])

print("  ✓ Computed: 1st/2nd serve %, ace/df rate, BP save %, return pts won %, dominance ratio")

# ── 5. Reshape to player-level rows ─────────────────────────────────────────
print("\n[5/6] Reshaping to player-level rows...")

# Create two copies: one from the winner perspective, one from the loser
winner_cols = {
    "winner_id": "player_id", "winner_name": "player_name", "winner_hand": "hand",
    "winner_ht": "height", "winner_ioc": "ioc", "winner_age": "age",
    "winner_rank": "rank", "winner_rank_points": "rank_points",
    "w_ace": "aces", "w_df": "dfs", "w_svpt": "serve_pts",
    "w_1stIn": "first_in", "w_1stWon": "first_won", "w_2ndWon": "second_won",
    "w_SvGms": "serve_games", "w_bpSaved": "bp_saved", "w_bpFaced": "bp_faced",
    "w_1st_serve_pct": "first_serve_pct", "w_1st_serve_win_pct": "first_serve_win_pct",
    "w_2nd_serve_win_pct": "second_serve_win_pct", "w_ace_rate": "ace_rate",
    "w_df_rate": "df_rate", "w_bp_save_pct": "bp_save_pct",
    "w_svc_pts_won": "service_pts_won_pct", "w_return_pts_won_pct": "return_pts_won_pct",
    "w_dominance_ratio": "dominance_ratio",
}
loser_cols = {
    "loser_id": "player_id", "loser_name": "player_name", "loser_hand": "hand",
    "loser_ht": "height", "loser_ioc": "ioc", "loser_age": "age",
    "loser_rank": "rank", "loser_rank_points": "rank_points",
    "l_ace": "aces", "l_df": "dfs", "l_svpt": "serve_pts",
    "l_1stIn": "first_in", "l_1stWon": "first_won", "l_2ndWon": "second_won",
    "l_SvGms": "serve_games", "l_bpSaved": "bp_saved", "l_bpFaced": "bp_faced",
    "l_1st_serve_pct": "first_serve_pct", "l_1st_serve_win_pct": "first_serve_win_pct",
    "l_2nd_serve_win_pct": "second_serve_win_pct", "l_ace_rate": "ace_rate",
    "l_df_rate": "df_rate", "l_bp_save_pct": "bp_save_pct",
    "l_svc_pts_won": "service_pts_won_pct", "l_return_pts_won_pct": "return_pts_won_pct",
    "l_dominance_ratio": "dominance_ratio",
}

common_cols = ["tourney_id", "tourney_name", "surface", "tourney_level", 
               "tourney_date", "score", "best_of", "round", "minutes", "year"]

df_w = df_matches[common_cols + list(winner_cols.keys())].rename(columns=winner_cols)
df_w["won"] = 1

df_l = df_matches[common_cols + list(loser_cols.keys())].rename(columns=loser_cols)
df_l["won"] = 0

df_player_matches = pd.concat([df_w, df_l], ignore_index=True)
print(f"  Total player-match rows: {len(df_player_matches):,}")

# ── 6. Aggregate per player / surface / year ─────────────────────────────────
print("\n[6/6] Aggregating player profiles...")

stat_cols = [
    "first_serve_pct", "first_serve_win_pct", "second_serve_win_pct",
    "ace_rate", "df_rate", "bp_save_pct", "service_pts_won_pct",
    "return_pts_won_pct", "dominance_ratio"
]

agg_dict = {col: "mean" for col in stat_cols}
agg_dict["won"] = ["sum", "count"]
agg_dict["aces"] = "sum"
agg_dict["dfs"] = "sum"
agg_dict["rank"] = "min"
agg_dict["rank_points"] = "max"
agg_dict["height"] = "first"
agg_dict["hand"] = "first"
agg_dict["ioc"] = "first"

def flatten_and_rename(df_agg):
    """Flatten MultiIndex columns and rename won_sum/won_count."""
    # Detect which first-level names have multiple aggregations
    from collections import Counter
    first_level_counts = Counter(a for a, b in df_agg.columns)
    
    new_cols = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            a, b = col
            if first_level_counts[a] > 1:
                # Multiple aggs on same column → always keep both parts
                new_cols.append(f"{a}_{b}")
            else:
                # Single agg → drop the agg name
                new_cols.append(a)
        else:
            new_cols.append(col)
    df_agg.columns = new_cols
    df_agg = df_agg.rename(columns={"won_sum": "wins", "won_count": "total_matches"})
    df_agg["win_pct"] = safe_div(df_agg["wins"].values, df_agg["total_matches"].values)
    return df_agg.reset_index()

# Career aggregation (overall)
df_career = flatten_and_rename(
    df_player_matches.groupby(["player_id", "player_name"]).agg(agg_dict)
)

# Per-surface aggregation
df_surface = flatten_and_rename(
    df_player_matches.groupby(["player_id", "player_name", "surface"]).agg(agg_dict)
)

# Per-year aggregation
df_yearly = flatten_and_rename(
    df_player_matches.groupby(["player_id", "player_name", "year"]).agg(agg_dict)
)

# ── Save outputs ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

df_matches.to_csv(os.path.join(OUTPUT_DIR, "master_matches_2005_2024.csv"), index=False)
print(f"  ✓ master_matches_2005_2024.csv        ({len(df_matches):,} rows)")

df_player_matches.to_csv(os.path.join(OUTPUT_DIR, "player_match_log.csv"), index=False)
print(f"  ✓ player_match_log.csv                ({len(df_player_matches):,} rows)")

df_career.to_csv(os.path.join(OUTPUT_DIR, "player_career_profiles.csv"), index=False)
print(f"  ✓ player_career_profiles.csv          ({len(df_career):,} players)")

df_surface.to_csv(os.path.join(OUTPUT_DIR, "player_surface_profiles.csv"), index=False)
print(f"  ✓ player_surface_profiles.csv         ({len(df_surface):,} rows)")

df_yearly.to_csv(os.path.join(OUTPUT_DIR, "player_yearly_profiles.csv"), index=False)
print(f"  ✓ player_yearly_profiles.csv          ({len(df_yearly):,} rows)")

# ── Identify top 50 players by peak ranking points ──────────────────────────
print("\n" + "=" * 60)
print("TOP 50 PLAYERS (by peak ranking points, 2005-2024)")
print("=" * 60)

top50 = df_career.nlargest(50, "rank_points")[
    ["player_id", "player_name", "hand", "height", "ioc",
     "total_matches", "wins", "win_pct", "rank", "rank_points",
     "first_serve_pct", "first_serve_win_pct", "second_serve_win_pct",
     "ace_rate", "df_rate", "bp_save_pct", "service_pts_won_pct",
     "return_pts_won_pct", "dominance_ratio"]
]

for i, row in top50.head(20).iterrows():
    print(f"  {row['player_name']:25s}  rank_pts={int(row['rank_points']):>6,}  "
          f"W-L={int(row['wins']):>4}-{int(row['total_matches'] - row['wins']):>4}  "
          f"win%={row['win_pct']:.1%}")

top50.to_csv(os.path.join(OUTPUT_DIR, "top50_players.csv"), index=False)
top50.to_json(os.path.join(OUTPUT_DIR, "top50_players.json"), orient="records", indent=2)
print(f"\n  ✓ top50_players.csv / .json saved")

print("\n✅ Phase 1 complete!")
