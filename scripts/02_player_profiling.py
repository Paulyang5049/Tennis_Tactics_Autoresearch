"""
Phase 2: Player Profiling & Clustering
========================================
Clusters ATP players into tactical archetypes using K-Means,
generates enriched profile data with cluster labels and per-surface splits.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")

print("=" * 60)
print("PHASE 2: PLAYER PROFILING & CLUSTERING")
print("=" * 60)

# ── 1. Load Phase 1 outputs ─────────────────────────────────────────────────
print("\n[1/5] Loading Phase 1 outputs...")
df_career = pd.read_csv(os.path.join(DATA_DIR, "player_career_profiles.csv"))
df_surface = pd.read_csv(os.path.join(DATA_DIR, "player_surface_profiles.csv"))
df_top50 = pd.read_csv(os.path.join(DATA_DIR, "top50_players.csv"))
top50_ids = set(df_top50["player_id"].values)
print(f"  Career profiles: {len(df_career):,} players")

# ── 2. Prepare clustering features ──────────────────────────────────────────
print("\n[2/5] Preparing clustering features...")
df_cluster = df_career[df_career["total_matches"] >= 30].copy()
print(f"  Players with 30+ matches: {len(df_cluster):,}")

feature_cols = [
    "first_serve_pct", "first_serve_win_pct", "second_serve_win_pct",
    "ace_rate", "df_rate", "bp_save_pct", "service_pts_won_pct",
    "return_pts_won_pct", "dominance_ratio"
]
df_cluster = df_cluster.dropna(subset=feature_cols)
print(f"  After dropping NaN: {len(df_cluster):,}")

X = df_cluster[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. K-Means clustering ───────────────────────────────────────────────────
print("\n[3/5] Running K-Means clustering...")
results = []
for k in range(3, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    results.append((k, score))
    print(f"  k={k}: silhouette={score:.4f}")

best_k = max(results, key=lambda x: x[1])[0]
print(f"\n  Best k = {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_cluster["cluster_kmeans"] = km_final.fit_predict(X_scaled)

# ── 4. Interpret clusters ───────────────────────────────────────────────────
print("\n[4/5] Interpreting cluster archetypes...")
cluster_profiles = df_cluster.groupby("cluster_kmeans")[feature_cols].mean()

archetype_names = {}
for cid in range(best_k):
    p = cluster_profiles.loc[cid]
    avg = cluster_profiles.mean()
    if p["ace_rate"] > avg["ace_rate"] * 1.15 and p["service_pts_won_pct"] > avg["service_pts_won_pct"]:
        name = "Big Servers"
    elif p["return_pts_won_pct"] > avg["return_pts_won_pct"] * 1.05 and p["second_serve_win_pct"] > avg["second_serve_win_pct"]:
        name = "Counterpunchers"
    elif p["service_pts_won_pct"] > avg["service_pts_won_pct"] and p["return_pts_won_pct"] > avg["return_pts_won_pct"]:
        name = "All-Court Elite"
    elif p["bp_save_pct"] > avg["bp_save_pct"] * 1.03:
        name = "Clutch Performers"
    elif p["dominance_ratio"] < avg["dominance_ratio"] * 0.95:
        name = "Grinders"
    else:
        name = f"Balanced (Tier {cid + 1})"
    archetype_names[cid] = name

df_cluster["archetype"] = df_cluster["cluster_kmeans"].map(archetype_names)

for cid in range(best_k):
    members = df_cluster[df_cluster["cluster_kmeans"] == cid]
    top_members = members.nlargest(5, "rank_points")["player_name"].tolist()
    in_top50 = members[members["player_id"].isin(top50_ids)]
    p = cluster_profiles.loc[cid]
    print(f"\n  Cluster {cid}: {archetype_names[cid]} ({len(members)} players)")
    print(f"    Top 50 members: {len(in_top50)}")
    print(f"    Examples: {', '.join(top_members[:5])}")
    print(f"    1stW%={p['first_serve_win_pct']:.1%}  Ace={p['ace_rate']:.1%}  Ret={p['return_pts_won_pct']:.1%}  BP%={p['bp_save_pct']:.1%}")

# ── 5. PCA & enriched profiles ──────────────────────────────────────────────
print("\n\n[5/5] Running PCA & building enriched profiles...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cluster["pca_x"] = X_pca[:, 0]
df_cluster["pca_y"] = X_pca[:, 1]
print(f"  PCA variance explained: {sum(pca.explained_variance_ratio_):.1%}")

# Build enriched top 50 profiles
top50_enriched = df_cluster[df_cluster["player_id"].isin(top50_ids)].copy()

# Merge surface data
for surface in ["Hard", "Clay", "Grass"]:
    sd = df_surface[df_surface["surface"] == surface][
        ["player_id", "total_matches", "wins", "win_pct",
         "first_serve_pct", "first_serve_win_pct", "second_serve_win_pct",
         "ace_rate", "return_pts_won_pct"]
    ].rename(columns=lambda c: f"{surface.lower()}_{c}" if c != "player_id" else c)
    top50_enriched = top50_enriched.merge(sd, on="player_id", how="left")

# Compute strengths/weaknesses
tour_avg = df_cluster[feature_cols].mean()

def get_sw(row):
    labels = {
        "first_serve_pct": "1st Serve %", "first_serve_win_pct": "1st Serve Win %",
        "second_serve_win_pct": "2nd Serve Win %", "ace_rate": "Ace Rate",
        "df_rate": "Double Fault Rate", "bp_save_pct": "Break Point Save %",
        "service_pts_won_pct": "Service Pts Won %", "return_pts_won_pct": "Return Pts Won %",
    }
    diffs = {}
    for col, lab in labels.items():
        if pd.notna(row[col]) and pd.notna(tour_avg[col]):
            diffs[lab] = -(row[col] - tour_avg[col]) if col == "df_rate" else (row[col] - tour_avg[col])
    sd = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    return sd[:3], sd[-3:]

profiles_json = []
for _, row in top50_enriched.iterrows():
    strengths, weaknesses = get_sw(row)
    profile = {
        "player_id": int(row["player_id"]),
        "player_name": row["player_name"],
        "hand": row.get("hand", ""), "height": int(row["height"]) if pd.notna(row.get("height")) else None,
        "ioc": row.get("ioc", ""), "archetype": row["archetype"],
        "cluster": int(row["cluster_kmeans"]),
        "pca_x": round(float(row["pca_x"]), 3), "pca_y": round(float(row["pca_y"]), 3),
        "career_stats": {
            "total_matches": int(row["total_matches"]), "wins": int(row["wins"]),
            "win_pct": round(float(row["win_pct"]), 4), "best_rank_points": int(row["rank_points"]),
        },
        "serve_profile": {k: round(float(row[k]), 4) for k in ["first_serve_pct","first_serve_win_pct","second_serve_win_pct","ace_rate","df_rate","bp_save_pct"]},
        "overall_profile": {k: round(float(row[k]), 4) for k in ["service_pts_won_pct","return_pts_won_pct","dominance_ratio"]},
        "strengths": [{"metric": s[0], "diff": f"+{s[1]:.1%}"} for s in strengths],
        "weaknesses": [{"metric": w[0], "diff": f"{w[1]:.1%}"} for w in weaknesses],
        "surface_splits": {}
    }
    for surface in ["hard", "clay", "grass"]:
        mc = f"{surface}_total_matches"
        if mc in row.index and pd.notna(row[mc]):
            profile["surface_splits"][surface] = {
                "matches": int(row[f"{surface}_total_matches"]), "wins": int(row[f"{surface}_wins"]),
                "win_pct": round(float(row[f"{surface}_win_pct"]), 4),
            }
    profiles_json.append(profile)

profiles_json.sort(key=lambda x: x["career_stats"]["best_rank_points"], reverse=True)

# ── Save ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

df_cluster.to_csv(os.path.join(DATA_DIR, "player_clusters.csv"), index=False)
print(f"  ✓ player_clusters.csv            ({len(df_cluster):,} players)")

cluster_profiles.to_csv(os.path.join(DATA_DIR, "cluster_centroids.csv"))
print(f"  ✓ cluster_centroids.csv          ({len(cluster_profiles)} clusters)")

with open(os.path.join(DATA_DIR, "archetype_map.json"), "w") as f:
    json.dump({str(k): v for k, v in archetype_names.items()}, f, indent=2)
print(f"  ✓ archetype_map.json")

with open(os.path.join(DATA_DIR, "top50_enriched_profiles.json"), "w") as f:
    json.dump(profiles_json, f, indent=2, default=str)
print(f"  ✓ top50_enriched_profiles.json   ({len(profiles_json)} players)")

# Sample
p = profiles_json[0]
print(f"\n  Sample: {p['player_name']} → {p['archetype']}")
print(f"  W-L: {p['career_stats']['wins']}-{p['career_stats']['total_matches']-p['career_stats']['wins']} ({p['career_stats']['win_pct']:.1%})")
print(f"  Strengths: {', '.join(s['metric'] for s in p['strengths'])}")
print(f"  Weaknesses: {', '.join(w['metric'] for w in p['weaknesses'])}")
surf_str = ', '.join(f'{s}: {d["win_pct"]:.0%}' for s, d in p['surface_splits'].items())
print(f"  Surfaces: {surf_str}")

print("\n✅ Phase 2 complete!")
