import pytest
import pandas as pd
import json
import tempfile
import os
from src.evaluation.evaluate import calculate_f05
from src.retrieval.candidate_generation import retrieve_candidates
import polars as pl

def test_calculate_f05():
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = os.path.join(tmpdir, "gt.tsv")
        pred_path = os.path.join(tmpdir, "pred.tsv")
        metrics_path = os.path.join(tmpdir, "metrics.json")
        
        gt = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1, S3-1", "", "S2-3"]
        })
        gt.to_csv(gt_path, sep="\t", index=False)
        
        pred = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1", "", "S2-3, S3-4"]
        })
        pred.to_csv(pred_path, sep="\t", index=False)
        
        calculate_f05(gt_path, pred_path, metrics_path)
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # S1-1: gt={S2-1, S3-1}, pred={S2-1}. TP=1, FP=0, FN=1. P=1.0, R=0.5. F0.5 = (1.25*1*0.5)/(0.25*1+0.5) = 0.625/0.75 = 0.833
        # S1-2: gt={}, pred={}. P=1.0, R=1.0. F0.5 = 1.0
        # S1-3: gt={S2-3}, pred={S2-3, S3-4}. TP=1, FP=1, FN=0. P=0.5, R=1.0. F0.5 = (1.25*0.5*1)/(0.25*0.5+1) = 0.625/1.125 = 0.555
        
        assert "macro_f05" in metrics
        assert metrics["macro_f05"] > 0
        
def test_candidate_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        s1 = pl.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "name_norm": ["acme", "globex"],
            "address_numbers": ["123", "456"],
            "country": ["US", "US"]
        })
        s2 = pl.DataFrame({
            "entity_id": ["S2-1", "S2-2"],
            "name_norm": ["acme", "other"],
            "address_numbers": ["123", "999"],
            "country": ["US", "US"]
        })
        s3 = pl.DataFrame({
            "entity_id": ["S3-1", "S3-2"],
            "name_norm": ["unknown", "globex"],
            "address_numbers": ["000", "456"],
            "country": ["US", "US"]
        })
        
        s1_path = os.path.join(tmpdir, "s1.parquet")
        s2_path = os.path.join(tmpdir, "s2.parquet")
        s3_path = os.path.join(tmpdir, "s3.parquet")
        out_path = os.path.join(tmpdir, "cands.parquet")
        
        s1.write_parquet(s1_path)
        s2.write_parquet(s2_path)
        s3.write_parquet(s3_path)
        
        retrieve_candidates(s1_path, s2_path, s3_path, out_path)
        
        cands = pl.read_parquet(out_path)
        # S1-1 should match S2-1 (name and address)
        # S1-2 should match S3-2 (name and address)
        
        assert cands.height == 2
        
        s1_1_cands = cands.filter(pl.col("source1_entity_id") == "S1-1")["candidate_entity_id"].to_list()
        assert "S2-1" in s1_1_cands
