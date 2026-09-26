import lightgbm as lgb
import polars as pl
import argparse
import json
import joblib
from pathlib import Path
import time

def train_model(train_features_path, val_features_path, model_output_path, metrics_path):
    print("Loading features...")
    df_train = pl.read_parquet(train_features_path)
    df_val = pl.read_parquet(val_features_path)
    
    feature_cols = [
        "exact_name_match", "exact_address_match", "country_match",
        "retrieved_by_name", "retrieved_by_address",
        "name_fuzz_ratio", "name_jaccard",
        "addr_fuzz_ratio", "addr_jaccard"
    ]
    
    X_train = df_train.select(feature_cols).to_pandas()
    y_train = df_train["label"].to_pandas()
    
    X_val = df_val.select(feature_cols).to_pandas()
    y_val = df_val["label"].to_pandas()
    
    print(f"Training on {len(X_train)} pairs, validating on {len(X_val)} pairs")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': 6,
        'n_jobs': -1,
        'random_state': 42
    }
    
    print("Training LightGBM...")
    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    print(f"Training finished in {time.time() - t0:.2f}s")
    
    # Save model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    imp_dict = {f: float(i) for f, i in zip(feature_cols, importance)}
    print("Feature importance (gain):", imp_dict)
    
    metrics = {
        "best_iteration": model.best_iteration,
        "feature_importance": imp_dict
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def predict(model_path, features_path, output_path):
    print("Loading model for prediction...")
    model = joblib.load(model_path)
    
    print("Loading features...")
    df = pl.read_parquet(features_path)
    
    feature_cols = [
        "exact_name_match", "exact_address_match", "country_match",
        "retrieved_by_name", "retrieved_by_address",
        "name_fuzz_ratio", "name_jaccard",
        "addr_fuzz_ratio", "addr_jaccard"
    ]
    
    X = df.select(feature_cols).to_pandas()
    
    print("Predicting probabilities...")
    preds = model.predict(X)
    
    df = df.with_columns(pl.Series("probability", preds, dtype=pl.Float32))
    
    # Just save the id and probability for decision engine
    df_out = df.select(["source1_entity_id", "candidate_entity_id", "probability"])
    df_out.write_parquet(output_path)
    print(f"Saved predictions to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--train_features', required=False)
    parser.add_argument('--val_features', required=False)
    parser.add_argument('--features', required=False)
    parser.add_argument('--model_output', required=True)
    parser.add_argument('--metrics', required=False)
    parser.add_argument('--predictions', required=False)
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args.train_features, args.val_features, args.model_output, args.metrics)
    elif args.mode == 'predict':
        predict(args.model_output, args.features, args.predictions)
