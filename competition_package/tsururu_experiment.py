import os

import pandas as pd


def run_univariate(df_raw, Pipeline, TSDataset, MLTrainer, KFoldCrossValidator, CatBoost, RecursiveStrategy) -> None:
    """
    Univariate Tsururu experiment on a single feature (baseline from earlier).
    """
    meta_cols = {"seq_ix", "step_in_seq", "need_prediction"}
    feature_cols = [c for c in df_raw.columns if c not in meta_cols]
    if not feature_cols:
        raise ValueError("No feature columns found in train.parquet")

    target_feature = feature_cols[0]
    print(f"\n=== Univariate experiment: target feature '{target_feature}' ===")

    df = df_raw[["seq_ix", "step_in_seq", target_feature]].copy()
    df["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        df["step_in_seq"], unit="D"
    )
    df.rename(columns={"seq_ix": "id", target_feature: "value"}, inplace=True)
    df = df[["id", "date", "value"]].sort_values(["id", "date"])

    print("\nMapped data sample (univariate):")
    print(df.head())

    dataset_params = {
        "target": {"columns": ["value"], "type": "continuous"},
        "date": {"columns": ["date"], "type": "datetime"},
        "id": {"columns": ["id"], "type": "categorical"},
    }

    dataset = TSDataset(
        data=df,
        columns_params=dataset_params,
        print_freq_period_info=True,
    )

    pipeline_easy_params = {
        "target_lags": 10,
        "date_lags": 1,
        "target_normalizer": "standard_scaler",
        "target_normalizer_regime": "none",
    }
    pipeline = Pipeline.easy_setup(
        dataset_params, pipeline_easy_params, multivariate=False
    )

    model = CatBoost
    model_params = {
        "loss_function": "RMSE",
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "early_stopping_rounds": 50,
        "verbose": 100,
    }

    validation = KFoldCrossValidator
    validation_params = {"n_splits": 3}

    trainer = MLTrainer(
        model=model,
        model_params=model_params,
        validator=validation,
        validation_params=validation_params,
    )

    horizon = 1
    history = 50
    strategy = RecursiveStrategy(horizon, history, trainer, pipeline)

    print("\nFitting RecursiveStrategy (univariate, horizon=1, history=50)...")
    fit_time, _ = strategy.fit(dataset)
    print(f"Univariate fit time (seconds): {fit_time:.3f}")


def run_multivariate(df_raw, Pipeline, TSDataset, MLTrainer, KFoldCrossValidator, CatBoost, RecursiveStrategy) -> None:
    """
    Multivariate Tsururu experiment: use all 32 features as a multivariate target.
    """
    meta_cols = {"seq_ix", "step_in_seq", "need_prediction"}
    feature_cols = [c for c in df_raw.columns if c not in meta_cols]
    if not feature_cols:
        raise ValueError("No feature columns found in train.parquet")

    print("\n=== Multivariate experiment: all feature columns as target ===")
    print(f"Number of target features: {len(feature_cols)}")

    df = df_raw[["seq_ix", "step_in_seq"] + feature_cols].copy()
    df["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        df["step_in_seq"], unit="D"
    )
    df.rename(columns={"seq_ix": "id"}, inplace=True)
    df = df[["id", "date"] + feature_cols].sort_values(["id", "date"])

    print("\nMapped data sample (multivariate):")
    print(df.head())

    dataset_params = {
        "target": {"columns": feature_cols, "type": "continuous"},
        "date": {"columns": ["date"], "type": "datetime"},
        "id": {"columns": ["id"], "type": "categorical"},
    }

    dataset = TSDataset(
        data=df,
        columns_params=dataset_params,
        print_freq_period_info=True,
    )

    pipeline_easy_params = {
        "target_lags": 10,
        "date_lags": 1,
        "target_normalizer": "standard_scaler",
        "target_normalizer_regime": "none",
    }
    # Note: we set multivariate=True to indicate multiple target columns.
    pipeline = Pipeline.easy_setup(
        dataset_params, pipeline_easy_params, multivariate=True
    )

    model = CatBoost
    model_params = {
        "loss_function": "MultiRMSE",
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "early_stopping_rounds": 50,
        "verbose": 100,
    }

    validation = KFoldCrossValidator
    validation_params = {"n_splits": 3}

    trainer = MLTrainer(
        model=model,
        model_params=model_params,
        validator=validation,
        validation_params=validation_params,
    )

    horizon = 1
    history = 50
    strategy = RecursiveStrategy(horizon, history, trainer, pipeline)

    print("\nFitting RecursiveStrategy (multivariate, horizon=1, history=50)...")
    fit_time, _ = strategy.fit(dataset)
    print(f"Multivariate fit time (seconds): {fit_time:.3f}")

    print("\nPredicting over dataset (for inspection)...")
    forecast_time, current_pred = strategy.predict(dataset)
    print(f"Multivariate forecast time (seconds): {forecast_time:.3f}")
    print("\nPredictions head (multivariate):")
    print(current_pred.head(10))


def main():
    """
    Offline experiments:
    - Univariate CatBoost + RecursiveStrategy (horizon=1) on one feature.
    - Multivariate CatBoost + RecursiveStrategy (horizon=1) on all 32 features.
    """
    try:
        from tsururu.dataset import Pipeline, TSDataset
        from tsururu.model_training.trainer import MLTrainer
        from tsururu.model_training.validator import KFoldCrossValidator
        from tsururu.models.boost import CatBoost
        from tsururu.strategies import RecursiveStrategy
    except ImportError:
        print(
            "Tsururu is not installed in this environment.\n"
            "Install it in your .venv, for example:\n"
            "  pip install -U tsururu[catboost]\n"
        )
        raise

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {data_path}")
    df_raw = pd.read_parquet(data_path)

    # Run univariate experiment (baseline)
    run_univariate(
        df_raw, Pipeline, TSDataset, MLTrainer, KFoldCrossValidator, CatBoost, RecursiveStrategy
    )

    # Run multivariate experiment (all 32 targets)
    run_multivariate(
        df_raw, Pipeline, TSDataset, MLTrainer, KFoldCrossValidator, CatBoost, RecursiveStrategy
    )


if __name__ == "__main__":
    main()
