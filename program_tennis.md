# autoresearch — Tennis Tactics Intelligence

This is an autonomous experiment loop to build **tactical intelligence for tennis athletes**. The goal is not just to predict who wins a point — it's to understand **WHY** they win and produce **actionable tactical recommendations** that update a player profile dashboard.

## The Big Picture

The ultimate output of this system feeds into a web dashboard showing each top player's:
- **Strengths & weaknesses** (what they do well, what to exploit)
- **Serve direction recommendations** per situation (key points, 1st/2nd serve)
- **Best shot combinations** (serve+1 patterns, rally strategies)
- **Physical endurance profile** (distance indicators, rally tolerance)

---

## Sport Analytics Principles

These principles should guide every experiment. The agent is not just tuning hyperparameters — it's building models that capture tennis-specific tactical knowledge.

### Principle 1: Key Point Performance Is What Separates Champions

Not all points are created equal. In tennis analytics, **key points** are the moments that swing matches:

| Score | Why It Matters |
|---|---|
| **30-30** | Next point creates either a break point or game point. The server's margin disappears. |
| **30-40 / 40-30** | Break point situations — the highest-pressure moments in tennis. |
| **40-40 (Deuce)** | Extended pressure. Mental resilience determines the outcome. |
| **0-40** | Triple break point — massive pressure on server. |
| **Tiebreak points** | Every point matters 2x; mini-break = enormous swing. |

**What the model should learn**: On key points, serve placement matters more than speed. Players change their patterns under pressure — some go for bigger serves, others play safer. The model should capture this and identify:
- Which serve direction (Wide/T/Body) has the highest win rate on key points for each situation
- Whether 1st serve vs 2nd serve behavior differs on key points (data shows: 59.9% vs 43.9% server win rate)
- Whether the player's key-point strategy matches what's statistically optimal

### Principle 2: Serve Direction Is the Most Controllable Tactical Decision

The serve is the only shot in tennis where the player has complete control. Available directions:

| Code | Direction | Tactical Use |
|---|---|---|
| **W** (Wide) | Pulls opponent off court | Opens up the court for serve+1 |
| **C** / **T** (Center/T) | Down the middle | Reduces return angle, good under pressure |
| **B** (Body) | Directly at the returner | Jams the opponent, limits return quality |
| **BC** (Body/Center) | Between body and center | Hybrid — hard to read |
| **BW** (Body/Wide) | Between body and wide | Hybrid |

**What the model should learn**: Different serve directions have different win rates depending on:
- **Court side** (deuce vs ad) — e.g., wide serve on deuce = slicing backhand, wide serve on ad = stretching forehand
- **1st vs 2nd serve** — 1st serves to body/wide are riskier but more rewarding
- **Opponent handedness** — wide serve on ad to a left-hander is very different from a right-hander
- **Score context** — elite players change patterns on break points (data shows they go for T serves more)

### Principle 3: Serve+1 Combinations Reveal Tactical Planning

The biggest advantage in modern tennis comes from planning beyond the serve:

1. **Wide Serve → Cross-Court Winner**: Classic pattern — pull opponent wide, hit behind them
2. **T Serve → Forehand Inside-Out**: Serve center, short return, attack with forehand
3. **Body Serve → Volley**: Jam the return, come in behind it
4. **Wide Serve → Drop Shot**: After pulling opponent deep and wide

**What the model should learn**: Point win rate conditioned on (serve_direction, next_shot_type). The model should identify which players excel at which serve+1 combos and recommend the highest-win-rate patterns.

### Principle 4: Rally Length Is a Physical and Tactical Indicator

From the data:
- Mean rally length: ~4 shots across all points
- Key points average ~4.2 shots (slightly longer — more careful)

**Distance run** is a powerful feature:
- Mean distance per point: ~16.1m (when data available, 98.5% of points)
- **If a player consistently runs more distance**, it means:
  - They're being moved around the court (potential weakness: court coverage)
  - OR they're athletic enough to retrieve (potential strength: endurance)
  
**What the model should learn**:
- Players who run significantly more distance than their opponent AND lose → they're being tactically outmaneuvered
- Players with high rally win % in long rallies (7+ shots) → counterpuncher archetype
- Players who win most points in short rallies (1-3 shots) → Big Server archetype

### Principle 5: Pressure Changes Everything

The same player makes different decisions under pressure:
- **Break point (facing)**: Tend to go for safer serves, more T serves
- **Break point (opportunity)**: Returner becomes more aggressive
- **Set point**: Higher error rate due to nerves
- **Match momentum**: Winning streaks amplify confidence → bigger serving

**What the model should learn**: Interaction between score context and every other feature. The model should capture that a player who wins 60% of all points might only win 45% of break points — that's a weakness the dashboard should surface.

---

## Setup

To set up a new experiment:

1. **Agree on a run tag**: e.g. `mar16`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read the files**:
   - `prepare_tennis.py` — fixed data pipeline. **Do not modify.**
   - `train_tennis.py` — the file you modify.
4. **Verify data**: `conda run -n ml python prepare_tennis.py --stats`
5. **Initialize results.tsv**: Header row only.
6. **Run baseline**: `conda run -n ml python train_tennis.py > run.log 2>&1`

## The Task

You modify `train_tennis.py` to build better models that **learn the sport analytics principles above**. This means:

1. **Point prediction accuracy** (val_accuracy) is the primary metric — but it's not the only goal
2. The model should also output **tactical insights** that the dashboard can use:
   - Serve direction win rates by score context
   - Serve+1 effectiveness matrices
   - Key-point performance differentials
   - Physical endurance indicators

### Data Available

**Primary: Grand Slam PBP (1.9M points, 2011–2024)**
- 65 columns including serve speed, direction, width, depth, rally count, distance run, momentum
- Temporal split: 2011-2022 train / 2023 val / 2024 test

**Secondary: Match Charting Project (1.2M charted points)**
- Shot-by-shot sequences: every shot's direction, type, outcome
- Can extract: serve+1 combos, rally patterns, shot selection by situation

**Features (29 engineered from PBP):**

| Feature Group | Features | Tactical Meaning |
|---|---|---|
| **Serve** | PointServer, ServeNumber, Speed_KMH, ServeDir_enc, ServeWidth_enc, ServeDepth_enc | Server's tactical decisions |
| **Return** | ReturnDepth_enc | Return quality |
| **Score** | P1Score, P2Score, ScoreDiff, SetNo, P1/P2GamesWon, GameDiff | Match context and pressure |
| **Momentum** | P1/P2Momentum, MomentumDiff, P1/P2PointsWon, PointsWonDiff | Match flow |
| **Rally** | RallyCount | Rally length, finishing ability |
| **Pressure** | P1/P2BreakPoint, IsBreakPoint, ScorePressure, IsTiebreak | High-pressure situations |
| **Physical** | P1/P2Distance, ServerAhead | Movement and court coverage |

### Ideas the Agent Should Explore (Guided by Sport Analytics)

1. **Situation-aware model**: Train separate heads or embeddings for different game situations:
   - Regular points vs key points vs break points vs tiebreaks
   - The model should learn *different weights* for different situations

2. **Serve direction prediction**: Add a secondary task — predict which serve direction wins given the situation. This creates **tactical recommendations**.

3. **Rally-length conditioning**: Split analysis by rally length:
   - Short (1-3): predominantly serve quality
   - Medium (4-6): serve+1 patterns
   - Long (7+): endurance, court coverage, shot selection

4. **Distance-based features**:
   - Running distance ratio (P1 vs P2) — who's being moved more
   - Distance per rally shot — efficiency of movement
   - Cumulative distance in the match — fatigue factor

5. **Key-point weighting**: Weight key points higher in the loss function:
   ```python
   # NOT JUST uniform BCE — weight key points 2-3x
   weight = torch.where(is_key_point, 2.5, 1.0)
   loss = F.binary_cross_entropy(output, target, weight=weight)
   ```

6. **Multi-task learning**: Train the model to simultaneously predict:
   - Who wins the point (primary task)
   - Optimal serve direction for the situation (auxiliary task)
   - Rally length category (auxiliary task)
   
   This forces the model to learn richer tactical representations.

7. **Player embeddings**: If match_id encodes player identity, learn player-specific embeddings that capture individual style. This enables per-player tactical profiles.

8. **Temporal attention**: Points in a match are not i.i.d. — momentum, fatigue, and tactical adjustments create temporal dependencies. The model should attend to recent history with different windows.

## Rules

**CAN modify**: `train_tennis.py` — architecture, hyperparameters, loss functions, features, outputs, everything.

**CANNOT modify**: `prepare_tennis.py` — fixed data loading, feature engineering, and evaluation metric.

**Run command**: `conda run -n ml python train_tennis.py`

**Time budget**: 5 minutes training time per experiment (wall clock).

**Primary metric**: `val_accuracy` (higher is better). Extract with: `grep "^val_accuracy:" run.log`

## Output Format

```
---
val_accuracy:     0.621700
val_auc:          0.654300
training_seconds: 300.1
total_seconds:    325.9
...
```

## Logging

Log to `results.tsv` (tab-separated):
```
commit	val_accuracy	val_auc	status	description
a1b2c3d	0.609000	0.631000	keep	baseline LSTM
...
```

## The Experiment Loop

LOOP FOREVER:
1. Check git state
2. Edit `train_tennis.py` with a sport-analytics-informed idea
3. `git commit`
4. `conda run -n ml python train_tennis.py > run.log 2>&1`
5. Read results: `grep "^val_accuracy:" run.log`
6. Record in TSV
7. If improved → keep. If not → reset.

**NEVER STOP.** Think about what the data is telling you about tennis tactics, not just loss curves.
