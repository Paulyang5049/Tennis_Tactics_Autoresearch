"""
Phase 3: Shot-Level Tactical Analysis
========================================
Analyzes Match Charting Project (MCP) data to baseline macro-level tour
trends (40-40 serve win%, rally length decay, serve directions) and extracts
individual player preferences for the top 50 players.
"""

import os
import json
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCP_DIR = os.path.join(BASE_DIR, "tennis_MatchChartingProject")
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")

print("=" * 60)
print("PHASE 3: SHOT-LEVEL TACTICAL ANALYSIS")
print("=" * 60)

# Load top 50 players
with open(os.path.join(DATA_DIR, "top50_enriched_profiles.json"), "r") as f:
    top50_profiles = json.load(f)

# The MCP files use player names that usually match the ATP data, but let's normalize
top50_names = [p["player_name"] for p in top50_profiles]
print(f"Loaded {len(top50_names)} top players to profile.")

# ── 1. Macro Tour Trends ────────────────────────────────────────────────────
print("\n[1/3] Computing Macro Tour Trends (Baseline)...")

# A. Key Points: Deuce / 40-40 Win %
df_kp = pd.read_csv(os.path.join(MCP_DIR, "charting-m-stats-KeyPointsServe.csv"), low_memory=False)
# Filter numeric pts just in case
df_kp = df_kp[pd.to_numeric(df_kp["pts"], errors="coerce").notnull()]
df_kp["pts"] = df_kp["pts"].astype(int)
df_kp["pts_won"] = df_kp["pts_won"].astype(int)

df_deuce = df_kp[df_kp["row"] == "Deuce"].copy()
tour_deuce_pts = df_deuce["pts"].sum()
tour_deuce_won = df_deuce["pts_won"].sum()
tour_deuce_win_pct = tour_deuce_won / tour_deuce_pts if tour_deuce_pts > 0 else 0

print(f"  Tour avg serve win% at Deuce: {tour_deuce_win_pct:.1%} ({tour_deuce_won:,}/{tour_deuce_pts:,} pts)")

# B. Rally Length Decay
df_rally = pd.read_csv(os.path.join(MCP_DIR, "charting-m-stats-Rally.csv"), low_memory=False)
df_rally = df_rally[df_rally["row"].isin(["1-3", "4-6", "7-9", "10"])]
df_rally = df_rally[pd.to_numeric(df_rally["pts"], errors="coerce").notnull()]
for col in ["pts", "pl1_won"]:
    df_rally[col] = df_rally[col].astype(int)

tour_rally_stats = df_rally.groupby("row")[["pts", "pl1_won"]].sum().reset_index()
tour_rally_stats["win_pct"] = tour_rally_stats["pl1_won"] / tour_rally_stats["pts"]
# Sort rows chronologically
row_order = {"1-3": 0, "4-6": 1, "7-9": 2, "10": 3}
tour_rally_stats["order"] = tour_rally_stats["row"].map(row_order)
tour_rally_stats = tour_rally_stats.sort_values("order")

print("  Tour average serve win% by Rally Length:")
for _, r in tour_rally_stats.iterrows():
    print(f"    Rally {r['row']:4s}: {r['win_pct']:.1%} ({r['pts']:,} pts)")

# C. Serve Direction Frequencies
df_dir = pd.read_csv(os.path.join(MCP_DIR, "charting-m-stats-ServeDirection.csv"), low_memory=False)
df_dir_total = df_dir[df_dir["row"] == "Total"].copy()
dir_cols = ["deuce_wide", "deuce_middle", "deuce_t", "ad_wide", "ad_middle", "ad_t"]
for col in dir_cols:
    df_dir_total[col] = pd.to_numeric(df_dir_total[col], errors="coerce").fillna(0).astype(int)

tour_dir_sums = df_dir_total[dir_cols].sum()
deuce_tot = tour_dir_sums["deuce_wide"] + tour_dir_sums["deuce_middle"] + tour_dir_sums["deuce_t"]
ad_tot = tour_dir_sums["ad_wide"] + tour_dir_sums["ad_middle"] + tour_dir_sums["ad_t"]

tour_serve_dir = {
    "deuce_wide_pct": float(tour_dir_sums["deuce_wide"] / deuce_tot),
    "deuce_middle_pct": float(tour_dir_sums["deuce_middle"] / deuce_tot),
    "deuce_t_pct": float(tour_dir_sums["deuce_t"] / deuce_tot),
    "ad_wide_pct": float(tour_dir_sums["ad_wide"] / ad_tot),
    "ad_middle_pct": float(tour_dir_sums["ad_middle"] / ad_tot),
    "ad_t_pct": float(tour_dir_sums["ad_t"] / ad_tot),
}

macro_trends = {
    "tour_deuce_serve_win_pct": float(tour_deuce_win_pct),
    "tour_rally_decay": {
        r["row"]: float(r["win_pct"]) for _, r in tour_rally_stats.iterrows()
    },
    "tour_serve_directions": tour_serve_dir
}

with open(os.path.join(DATA_DIR, "tour_macro_trends.json"), "w") as f:
    json.dump(macro_trends, f, indent=2)

# ── 2. Top 50 Individual Player Match Charting Extraction ───────────────────
print("\n[2/3] Extracting MCP Profiles for Top 50 Players...")

player_mcp_stats = {}

# Aggregate player data
# Deuce points
player_deuce = df_deuce[df_deuce["player"].isin(top50_names)].groupby("player")[["pts", "pts_won"]].sum()
# Rally points
player_rally = df_rally[df_rally["server"].isin(top50_names)].groupby(["server", "row"])[["pts", "pl1_won"]].sum().reset_index()
# Serve direction
player_dir = df_dir_total[df_dir_total["player"].isin(top50_names)].groupby("player")[dir_cols].sum()

for player_name in top50_names:
    stats = {
        "mcp_data_exists": False,
        "deuce_win_pct": None,
        "rally_win_pcts": {"1-3": None, "4-6": None, "7-9": None, "10": None},
        "serve_directions": {}
    }
    
    # 1. Deuce Win %
    if player_name in player_deuce.index:
        row = player_deuce.loc[player_name]
        if row["pts"] > 10:
            stats["mcp_data_exists"] = True
            stats["deuce_win_pct"] = float(row["pts_won"] / row["pts"])
            
    # 2. Rally Win %
    pr = player_rally[player_rally["server"] == player_name]
    for _, r in pr.iterrows():
        if r["pts"] > 10:
            stats["mcp_data_exists"] = True
            stats["rally_win_pcts"][r["row"]] = float(r["pl1_won"] / r["pts"])
            
    # 3. Serve Directions
    if player_name in player_dir.index:
        p_dir = player_dir.loc[player_name]
        p_deuce_tot = p_dir["deuce_wide"] + p_dir["deuce_middle"] + p_dir["deuce_t"]
        p_ad_tot = p_dir["ad_wide"] + p_dir["ad_middle"] + p_dir["ad_t"]
        
        if p_deuce_tot > 20 and p_ad_tot > 20:
            stats["mcp_data_exists"] = True
            stats["serve_directions"] = {
                "deuce_wide_pct": float(p_dir["deuce_wide"] / p_deuce_tot),
                "deuce_t_pct": float(p_dir["deuce_t"] / p_deuce_tot),
                "ad_wide_pct": float(p_dir["ad_wide"] / p_ad_tot),
                "ad_t_pct": float(p_dir["ad_t"] / p_ad_tot),
            }
            
    player_mcp_stats[player_name] = stats

# Enlist into main profiles JSON or save as separate
with open(os.path.join(DATA_DIR, "top50_mcp_tactical_stats.json"), "w") as f:
    json.dump(player_mcp_stats, f, indent=2)

players_with_data = sum(1 for s in player_mcp_stats.values() if s["mcp_data_exists"])
print(f"  Extracted charting stats for {players_with_data}/{len(top50_names)} top players.")

# ── 3. Append to existing profiles ──────────────────────────────────────────
print("\n[3/3] Appending tactical stats to enriched profiles...")
for p in top50_profiles:
    name = p["player_name"]
    if name in player_mcp_stats:
        p["mcp_tactical_profile"] = player_mcp_stats[name]

with open(os.path.join(DATA_DIR, "top50_enriched_profiles.json"), "w") as f:
    json.dump(top50_profiles, f, indent=2)

print("  ✓ Updated top50_enriched_profiles.json")

print("\n✅ Phase 3 complete!")
