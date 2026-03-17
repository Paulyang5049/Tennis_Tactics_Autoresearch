# Autoresearch for Tennis Tactics — Repo Intro & Implementation Plan

## Repo Introduction

This repo ([miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)) is a fork of Karpathy's **autoresearch** project, adapted for Apple Silicon Macs. The core idea: an AI agent autonomously iterates on a training script — modifying architecture/hyperparameters, running 5-minute experiments, keeping improvements, discarding failures — all without human intervention.

### Repo Structure

| Component | Purpose |
|---|---|
| [train.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/train.py) | **Agent-editable.** GPT model + MuonAdamW optimizer + training loop. Currently trains a language model on text. |
| [prepare.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/prepare.py) | **Read-only.** Data prep, tokenizer, dataloader, [evaluate_bpb](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/prepare.py#356-380) metric. |
| [program.md](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/program.md) | **Agent instructions.** Defines the autonomous experiment loop. |
| `tennis_atp/` | 56 years of ATP match CSVs (1968–2024), player bios, rankings — *freshly cloned* |
| `tennis_MatchChartingProject/` | Shot-level charting: serve directions, rally stats, shot outcomes — *freshly cloned* |
| `tennis_slam_pointbypoint/` | Grand Slam point-by-point data (2011–2024) — *freshly cloned* |
| `dashboard_data/` | Pre-processed outputs from your existing pipeline |
| `scripts/` | Your tennis pipeline (Phases 1–4, 6) |

### Your Existing Tennis Pipeline

| Script | Phase | What it does |
|---|---|---|
| [01_ingest_and_clean.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/scripts/01_ingest_and_clean.py) | Ingest | Merges ATP 2005–2024, computes serve/return metrics, identifies top 50 |
| [02_player_profiling.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/scripts/02_player_profiling.py) | Profiling | K-Means clustering into archetypes (Big Servers, Counterpunchers, etc.) |
| [03_shot_level_analysis.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/scripts/03_shot_level_analysis.py) | Tactics | MCP shot-level: deuce win%, rally decay, serve directions |
| [04_deep_learning_models.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/scripts/04_deep_learning_models.py) | **Models** | LSTM, Transformer, Autoencoder, HistGradientBoosting for point prediction |
| [06_extract_top_shots_and_combos.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/scripts/06_extract_top_shots_and_combos.py) | Combos | Top winning shots and serve+1 combos per player |

---

## 🔴 Data Analysis: The Model Desperately Needs More Data

> [!CAUTION]
> **Your current model uses only ~14,600 points from 2 tournaments.** With 3.1 million points available, the model is starved of data — explaining the mediocre 53–62% accuracy.

### Data Inventory (Freshly Cloned Repos)

| Data Source | Points/Rows | Currently Used | Utilization |
|---|---:|---:|---:|
| **Grand Slam PBP** (`tennis_slam_pointbypoint/`) | **1,922,185** | ~14,600 (2 tournaments) | **0.8%** |
| **MCP Charting Points** (`tennis_MatchChartingProject/`) | **1,221,865** | 0 | **0%** |
| **ATP Match Stats** (`dashboard_data/master_matches`) | **53,857** | 0 (used in scripts 01–03 only) | **0%** |
| **Total available** | **~3.2M points** | **~14,600** | **0.5%** |

### Why More Data Will Dramatically Improve the Model

1. **130x more training data** — From 14,600 → 1.9M Grand Slam points alone. Deep learning models are data-hungry; the current LSTM/Transformer are severely undertrained.

2. **Richer feature space from MCP data** — The 1.2M MCP points include shot-by-shot sequences (forehand/backhand, directions, shot types) that the current model doesn't use at all. This enables modeling rally patterns rather than just serve + score features.

3. **Cross-tournament generalization** — Training on 2 tournaments means the model memorizes player matchup artifacts. With 49+ tournament files spanning 2011–2024, it must learn actual tactical patterns.

4. **The HistGradientBoosting 98.6% accuracy is a red flag** — likely overfitting to serve/score features that trivially correlate with outcomes in small data. More data will expose this and force the model to learn real tactics.

### Recommended Data Strategy for Autoresearch

| Phase | Data | Points | Purpose |
|---|---|---:|---|
| **Training** | All PBP 2011–2022 (36 files) | ~1.4M | Main training corpus |
| **Validation** | PBP 2023 (4 files) | ~200K | Metric for autoresearch keep/discard |
| **Test** | PBP 2024 (2 files) | ~80K | Final holdout, never used during autoresearch |
| **Feature enrichment** | MCP charting + ATP stats | ~1.2M + 54K | Additional features to engineer |

---

## Proposed Changes (Updated)

### New Files

#### [NEW] `prepare_tennis.py` — Fixed data pipeline (agent cannot modify)
- Loads **all** Grand Slam PBP data (2011–2024), not just 2 tournaments
- Temporal train/val/test split (2011–2022 / 2023 / 2024)
- Feature engineering from PBP + optional MCP enrichment
- Fixed `evaluate_accuracy(model, val_loader)` metric
- Uses conda [ml](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/pyproject.toml) environment

#### [NEW] `train_tennis.py` — Agent-editable model (seeded from [04_deep_learning_models.py](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/scripts/04_deep_learning_models.py))
- LSTM, Transformer, and hybrid architectures
- Hyperparameters, features, sequence length all fair game

#### [NEW] `program_tennis.md` — Agent instructions for tennis experiments
- Metric: validation accuracy (higher is better)
- Run command: `conda run -n ml python train_tennis.py`

---

## Verification Plan

### Automated Tests
1. Verify conda [ml](file:///Users/yangpaul/Desktop/Tennis%20training/autoresearch-macos/pyproject.toml) env has PyTorch + MPS support
2. Verify data loading from freshly cloned repos
3. Run baseline training, confirm `results.tsv` is created

### Manual Verification
- After baseline, confirm autonomous loop can be kicked off via `program_tennis.md`
