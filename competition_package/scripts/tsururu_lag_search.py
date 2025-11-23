

import os
import pandas as pd
import numpy as np

def run_search():
    try:
        from tsururu.dataset import Pipeline, TSDataset
        from tsururu.model_training.trainer import MLTrainer
        from tsururu.model_training.validator import KFoldCrossValidator
        from tsururu.models.boost import CatBoost
        from tsururu.strategies import RecursiveStrategy
    except ImportError:
        print("Tsururu not found. Install with `pip install -U tsururu[catboost]`.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "datasets", "train.parquet")
    
    print(f"Loading data from {data_path}...")
    df_raw = pd.read_parquet(data_path)
    
    meta_cols = {"seq_ix", "step_in_seq", "need_prediction"}
    feature_cols = [c for c in df_raw.columns if c not in meta_cols]
    
    # Subsample sequences FIRST to keep size manageable
    all_seqs = df_raw["seq_ix"].unique()
    rng = np.random.default_rng(42)
    subset_seqs = rng.choice(all_seqs, size=int(len(all_seqs) * 0.10), replace=False) # 10% (~50 seqs)
    df_sub = df_raw[df_raw["seq_ix"].isin(subset_seqs)].copy()
    
    print(f"Subsampled to {len(subset_seqs)} sequences. Melting to long format...")
    
    # Melt to Long Format: id = "{seq_ix}_{feature}"
    # We need to preserve step_in_seq for date
    df_melt = df_sub.melt(
        id_vars=["seq_ix", "step_in_seq"], 
        value_vars=feature_cols, 
        var_name="feature", 
        value_name="value"
    )
    
    # Create composite ID
    df_melt["id"] = df_melt["seq_ix"].astype(str) + "_" + df_melt["feature"].astype(str)
    
    # Create Date
    df_melt["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df_melt["step_in_seq"], unit="D")
    
    # Select columns
    df_final = df_melt[["id", "date", "value"]].sort_values(["id", "date"])
    
    print(f"Final dataset shape: {df_final.shape}")
    print(df_final.head())

    dataset_params = {
        "target": {"columns": ["value"], "type": "continuous"},
        "date": {"columns": ["date"], "type": "datetime"},
        "id": {"columns": ["id"], "type": "categorical"},
    }

    dataset = TSDataset(
        data=df_final,
        columns_params=dataset_params,
        print_freq_period_info=False,
    )

    lags_to_test = [10, 20, 30, 50]
    normalizers_to_test = ["standard_scaler", "difference"]
    
    results = []

    for norm in normalizers_to_test:
        for n_lags in lags_to_test:
            print(f"\n--- Testing Lags={n_lags}, Normalizer={norm} ---")
            
            pipeline_params = {
                "target_lags": n_lags,
                "date_lags": 1,
                "target_normalizer": norm,
                "target_normalizer_regime": "none" if norm == "standard_scaler" else "common",
            }
            
            try:
                # Multivariate=False because we now have a single target column in long format
                pipeline = Pipeline.easy_setup(
                    dataset_params, pipeline_params, multivariate=False
                )
                
                model_params = {
                    "loss_function": "RMSE",
                    "iterations": 150, # Fast search
                    "learning_rate": 0.1,
                    "depth": 6,
                    "early_stopping_rounds": 20,
                    "verbose": 0,
                    "thread_count": 4
                }
                
                trainer = MLTrainer(
                    model=CatBoost,
                    model_params=model_params,
                    validator=KFoldCrossValidator,
                    validation_params={"n_splits": 3},
                )
                
                # History needs to be longer than lags
                strategy = RecursiveStrategy(horizon=1, history=n_lags + 10, trainer=trainer, pipeline=pipeline)
                
                fit_time, metrics = strategy.fit(dataset)
                
                print(f"Fit Time: {fit_time:.2f}s. Metrics: {metrics}")
                results.append({
                    "lags": n_lags,
                    "norm": norm,
                    "score": metrics,
                    "time": fit_time
                })
                
            except Exception as e:
                print(f"Experiment failed: {e}")
                import traceback
                traceback.print_exc()

    print("\n=== Final Results ===")
    res_df = pd.DataFrame(results)
    print(res_df)
    
    res_df.to_csv(os.path.join(base_dir, "tsururu_lag_search_results.csv"), index=False)

if __name__ == "__main__":
    run_search()