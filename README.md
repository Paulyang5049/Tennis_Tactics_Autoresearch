# Tennis Tactics Autoresearch

An autonomous research system applied to tennis match data.

## Project Overview

This project embodies a system to analyze and discover tennis tactics using autonomous AI agents. Drawing inspiration from `autoresearch` architectures, we empower agents to dive into tennis match charting data, identify key-point performance, serve direction effectiveness, and optimal serve+1 combinations for top ATP players. 

The ultimate goal is to produce a detailed player profile dashboard outlining the strengths, weaknesses, and training recommendations for professional tennis players, driven by fully autonomous data analysis and feature extraction experiments.

## Features

- **Autonomous Experimentation**: An agent autonomously edits scripts to extract and test new tactical features from tennis datasets.
- **Match Charting Data Integration**: Parses and processes data from the Tennis Match Charting Project.
- **Tactical Modeling**: Iterates on models predicting shot outcomes, rally lengths, and match dynamics.
- **Player Profiling**: Maps the "Top Shots & Combos" of the top 50 players.

## Repository Contents

- `prepare_tennis.py` / `prepare.py`: Scripts handling data ingestion (e.g., Match Charting Project data).
- `train_tennis.py` / `train.py`: The core scripts modified by the AI agents during the autoresearch loop to experiment with different tactical models.
- `program.md`: Instructions defining the autonomous agent's goals and limits.
- `tennis_MatchChartingProject`, `tennis_atp`, `tennis_slam_pointbypoint`: Assorted raw data sources used for analysis.

## Getting Started

Make sure you have `uv` installed to manage dependencies.

```bash
uv sync
```

To run a single testing experiment on the data:
```bash
uv run train_tennis.py
```
