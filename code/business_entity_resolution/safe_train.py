import polars as pl
import time
import gc
import json
import duckdb
import lightgbm as lgb
import joblib

FEATURE_COLS = [
    "exact_name_match", "exact_legal_match", "sorted_name_match",
    "exact_address_match", "country_match", "addr_nums_match",
    "name_len_diff", "name_len_ratio", "addr_len_diff",
    "retrieved_by_name", "retrieved_by_addr", "retrieved_by_punct",
    "retrieved_by_token", "retrieved_by_bigram", "retrieved_by_phonetic",
    "retrieved_by_sorted", "retrieved_by_postal", "num_channels",
    "name_jaccard", "name_containment", "name_shared_tokens",
    "addr_jaccard", "addr_containment", "addr_shared_tokens",
    "name_fuzz_ratio", "name_partial_ratio", "name_jaro_winkler",
    "addr_fuzz_ratio",
]

def train_lgbm_safe(train_features_path, val_features_path, model_output_path, metrics_path):
    print("=" * 60)
    print("LIGHTGBM TRAINING (DuckDB hard negative sampling)")
    print("=" * 60)
    t0 = time.time()

    con = duckdb.connect()
    con.execute("PRAGMA max_temp_directory_size='20GiB'")

    train_chunks = str(train_features_path).replace(".parquet", "_chunk_*.parquet")
    val_chunks = str(val_features_path).replace(".parquet", "_chunk_*.parquet")

    print("Sampling training set via DuckDB...")
    # Fetch all positives
    con.execute(f"CREATE TABLE pos_train AS SELECT * FROM read_parquet('{train_chunks}') WHERE label = 1")
    pos_count = con.execute("SELECT COUNT(*) FROM pos_train").fetchone()[0]
    
    # Fetch hard negatives
    con.execute(f"""
        CREATE TABLE hard_neg AS 
        SELECT * FROM read_parquet('{train_chunks}') 
        WHERE label = 0 AND (
            name_fuzz_ratio > 0.3 OR 
            name_jaccard > 0.2 OR 
            exact_address_match = 1 OR 
            addr_jaccard > 0.3 OR 
            num_channels >= 2
        )
    """)
    hard_neg_count = con.execute("SELECT COUNT(*) FROM hard_neg").fetchone()[0]
    
    # Fetch random negatives
    max_random_neg = min(pos_count * 2, 5_000_000)
    con.execute(f"""
        CREATE TABLE rand_neg AS 
        SELECT * FROM read_parquet('{train_chunks}') 
        WHERE label = 0 USING SAMPLE {max_random_neg}
    """)
    
    print(f"  Train positives: {pos_count:,}")
    print(f"  Train hard negatives: {hard_neg_count:,}")

    # Combine in DuckDB
    con.execute("""
        CREATE TABLE train_balanced AS 
        SELECT * FROM pos_train
        UNION 
        SELECT * FROM hard_neg
        UNION 
        SELECT * FROM rand_neg
    """)

    # Fetch into Polars to convert to Pandas
    train_df = con.execute("SELECT * FROM train_balanced").pl()
    # deduplicate by entity pairs just in case random and hard overlap
    train_df = train_df.unique(subset=["source1_entity_id", "candidate_entity_id"])
    
    print(f"  Final training size: {train_df.height:,}")

    val_df = con.execute(f"SELECT * FROM read_parquet('{val_chunks}')").pl()
    print(f"  Validation size: {val_df.height:,}")

    con.close()

    X_train = train_df.select(FEATURE_COLS).to_pandas()
    y_train = train_df["label"].to_pandas()

    X_val = val_df.select(FEATURE_COLS).to_pandas()
    y_val = val_df["label"].to_pandas()

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
    }

    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(50),
        ]
    )
    
    train_time = time.time() - t0
    joblib.dump(model, model_output_path)
    
    importance = model.feature_importance(importance_type='gain')
    imp_dict = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
    
    metrics = {
        "best_iteration": model.best_iteration,
        "train_time": train_time,
        "feature_importance": {f: float(i) for f, i in imp_dict},
        "train_size": train_df.height,
        "pos_count": int(pos_count),
        "neg_count": int(neg_count),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    del train_df, val_df, X_train, y_train, X_val, y_val, train_data, val_data
    gc.collect()
    return model
