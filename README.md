# Market State Forecasting Challenge

This repository contains my work on a market-state sequence forecasting challenge.
The task, evaluator, streaming submission interface, and R2 leaderboard metric were provided by the competition.

I experimented with lag-based MLPs, residual blends, CatBoost, LSTM/GRU baselines, and early Mamba/SSM-style models.
The goal was to maximize leaderboard score under the provided step-by-step inference protocol.

Best documented result: LB 0.3571 for the level + residual blend.

## What To Look At

- [competition_package/README.md](competition_package/README.md) - challenge description, data format, evaluation, model notes, and packaging instructions.
- [competition_package/experiments/EXPERIMENT_LOG.md](competition_package/experiments/EXPERIMENT_LOG.md) - experiment history and validation notes.
- [competition_package/solution.py](competition_package/solution.py) - current submission entry point.
- [competition_package/scripts](competition_package/scripts) - training, validation, diagnostics, and model comparison scripts.
- [competition_package/src](competition_package/src) - reusable feature and model code.

## Repository Focus

This is a competition project rather than a production trading system. The main focus is practical score optimization on anonymized market-state sequences: validation design, model comparison, streaming inference, and submission packaging.
