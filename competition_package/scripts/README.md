# Scripts

Research and validation scripts for the market-state forecasting challenge.

## Common Entry Points

- `train_model.py` - main lag-MLP training pipeline.
- `train_mlp_v21.py` - later MLP training variant.
- `optimize_vector_blend.py` - blend search for level and residual models.
- `feature_consistency_check.py` - checks that offline and streaming features match.
- `leakage_check.py` and `leak_probe.py` - validation leakage diagnostics.

## Model Experiments

- `train_catboost_experiment.py`, `optimize_catboost.py`, `dump_catboost_importance.py`
- `train_lstm_experiment.py`, `train_feature_gru.py`
- `train_mamba_experiment.py`
- `train_tsmixer_experiment.py`, `train_tsmixer_refined.py`, `train_tsmixer_v5_full.py`
- `train_regime_classifier.py`, `cluster_regimes.py`, `evaluate_regimes.py`

## Analysis

- `scan_correlations.py`, `eda_returns.py`
- `compute_catch22_features.py`, `catch22_feature_importance.py`
- `adversarial_validation.py`, `calibrate_validation.py`
- `diagnostic_step_splits.py`

Most scripts assume they are run from `competition_package/` and that local data/model artifacts exist under ignored `datasets/` and `models/` directories.
