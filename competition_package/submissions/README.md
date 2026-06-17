# Submission Variants

This directory stores historical streaming-inference entry points used for leaderboard and local validation experiments.

Tracked Python files are kept as reference implementations. Packaged zip files and generated submission folders are ignored to keep the repository lightweight.

Useful variants:

- `solution_blend.py` - level + residual MLP blend, best documented leaderboard result.
- `solution_v21.py`, `solution_v22.py` - MLP variants.
- `solution_v23_gru.py` - GRU submission experiment.
- `solution_v24_scalar.py`, `solution_v25_regime.py` - scalar/regime variants.
- `solution_lstm.py` - LSTM baseline submission path.
- `solution_catboost.py` - CatBoost streaming experiment.

For an actual competition package, copy the desired variant to `../solution.py` and include the matching model artifacts.
