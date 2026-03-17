import pandas as pd
import json
import re
import os

print("Loading Top 50 enriched profiles...")
profiles_path = '/Users/yangpaul/Tennis_data/dashboard_data/top50_enriched_profiles.json'
with open(profiles_path, 'r') as f:
    profiles = json.load(f)

# Extract player names from keys or from player attribute
top50_names = [p['player_name'] for p in profiles]
print(f"Loaded {len(top50_names)} players.")

# Part 1: Top Shots from pre-aggregated data
print("Processing Best Shots from ShotDirOutcomes...")
df_shots = pd.read_csv('/Users/yangpaul/Tennis_data/tennis_MatchChartingProject/charting-m-stats-ShotDirOutcomes.csv')
df_shots = df_shots[df_shots['player'].isin(top50_names)]

shot_agg = df_shots.groupby(['player', 'row']).agg({
    'shots': 'sum',
    'shots_in_pts_won': 'sum',
    'winners': 'sum',
    'unforced': 'sum'
}).reset_index()

shot_agg['win_pct'] = shot_agg['shots_in_pts_won'] / shot_agg['shots']
shot_agg['winner_pct'] = shot_agg['winners'] / shot_agg['shots']
shot_agg['ue_pct'] = shot_agg['unforced'] / shot_agg['shots']

mapping = {
    'F-XC': 'Forehand Crosscourt',
    'F-DTM': 'Forehand Middle',
    'F-DTL': 'Forehand Line',
    'F-IO': 'Forehand Inside-Out',
    'F-II': 'Forehand Inside-In',
    'B-XC': 'Backhand Crosscourt',
    'B-DTM': 'Backhand Middle',
    'B-DTL': 'Backhand Line',
    'B-IO': 'Backhand Inside-Out',
    'B-II': 'Backhand Inside-In'
}

shot_agg = shot_agg[shot_agg['row'].isin(mapping.keys())]
shot_agg['shot_name'] = shot_agg['row'].map(mapping)

best_shots_per_player = {}
for player in top50_names:
    p_shots = shot_agg[(shot_agg['player'] == player) & (shot_agg['shots'] >= 100)]
    if p_shots.empty:
        p_shots = shot_agg[(shot_agg['player'] == player) & (shot_agg['shots'] >= 20)]
    
    top_shots = p_shots.sort_values('win_pct', ascending=False).head(3)
    best_shots_per_player[player] = top_shots[['shot_name', 'win_pct', 'shots', 'winner_pct']].to_dict('records')

# Part 2: Top Combos (Serve + 1)
print("Processing Serve+1 Combos from point sequences...")
df_20s = pd.read_csv('/Users/yangpaul/Tennis_data/tennis_MatchChartingProject/charting-m-points-2020s.csv', low_memory=False)
df_10s = pd.read_csv('/Users/yangpaul/Tennis_data/tennis_MatchChartingProject/charting-m-points-2010s.csv', low_memory=False)
df_pts = pd.concat([df_20s, df_10s], ignore_index=True)

def extract_serve_plus_1(row):
    seq = str(row['1st'])
    if pd.isna(seq) or len(seq) < 3:
        return None
        
    serve_char = seq[0]
    serve_dir = 'Wide' if serve_char == '4' else ('Body' if serve_char == '5' else ('T' if serve_char == '6' else None))
    
    if not serve_dir:
        return None
        
    clean_seq = re.sub(r'[*#@!+\-=\^x;]', '', seq)
    letters = re.findall(r'[a-zA-Z]', clean_seq)
    
    if len(letters) >= 2:
        plus_1_char = letters[1].lower()
        if plus_1_char == 'f': plus_1 = 'Forehand'
        elif plus_1_char == 'b': plus_1 = 'Backhand'
        elif plus_1_char in ['v', 'h', 'l', 'o', 'z']: plus_1 = 'Volley/Overhead'
        else: return None
        return f"{serve_dir} Serve + {plus_1}"
    return None

df_pts['serve_plus_1'] = df_pts.apply(extract_serve_plus_1, axis=1)
df_pts['isSvrWinner'] = df_pts.apply(lambda x: 1 if (str(x['PtWinner']) == str(x['Svr'])) else 0, axis=1)

def get_server_name(row):
    match = str(row['match_id'])
    parts = match.split('-')
    if len(parts) >= 6:
        p1 = parts[-2].replace('_', ' ')
        p2 = parts[-1].replace('_', ' ')
        return p1 if str(row['Svr']) == '1' else p2
    return None

df_pts['server_name'] = df_pts.apply(get_server_name, axis=1)
valid_pts = df_pts.dropna(subset=['serve_plus_1', 'server_name'])

combo_agg = valid_pts.groupby(['server_name', 'serve_plus_1']).agg(
    pts_played=('isSvrWinner', 'count'),
    pts_won=('isSvrWinner', 'sum')
).reset_index()

combo_agg['win_pct'] = combo_agg['pts_won'] / combo_agg['pts_played']

best_combos_per_player = {}
for player in top50_names:
    p_combos = combo_agg[(combo_agg['server_name'] == player) & (combo_agg['pts_played'] >= 30)]
    if p_combos.empty:
        p_combos = combo_agg[(combo_agg['server_name'] == player) & (combo_agg['pts_played'] >= 10)]
    
    top_combos = p_combos.sort_values('win_pct', ascending=False).head(3)
    best_combos_per_player[player] = top_combos[['serve_plus_1', 'win_pct', 'pts_played']].to_dict('records')

# Update enriched profiles
for p in profiles:
    player = p['player_name']
    p['top_shots'] = best_shots_per_player.get(player, [])
    p['top_combos'] = best_combos_per_player.get(player, [])

with open(profiles_path, 'w') as f:
    json.dump(profiles, f, indent=2)
    
print("Successfully enriched top50_enriched_profiles.json with top shots and combos!")
