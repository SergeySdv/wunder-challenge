# Market State Forecasting Challenge

This repository contains my work on a market-state sequence forecasting challenge.
The task is to predict the next market state vector from previous states using anonymized numeric features.

I experimented with lag-based MLPs, residual blends, CatBoost, LSTM/GRU baselines, and early Mamba/SSM-style models.
The project includes local validation, streaming-style inference, submission packaging, and R2-based evaluation.

Best documented result: LB 0.3571 for the level + residual blend.

## What To Look At

- [competition_package/README.md](competition_package/README.md) - challenge description, data format, evaluation, model notes, and packaging instructions.
- [competition_package/experiments/EXPERIMENT_LOG.md](competition_package/experiments/EXPERIMENT_LOG.md) - experiment history and validation notes.
- [competition_package/solution.py](competition_package/solution.py) - current submission entry point.
- [competition_package/scripts](competition_package/scripts) - training, validation, diagnostics, and model comparison scripts.
- [competition_package/src](competition_package/src) - reusable feature and model code.

## Repository Focus

This is a research-oriented project rather than a production trading system. The main focus is practical market-data experimentation: sequence validation, noisy time-series forecasting, reproducible model comparison, and competition-style inference packaging.
