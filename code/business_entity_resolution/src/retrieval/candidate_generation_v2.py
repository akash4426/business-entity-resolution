import polars as pl
import argparse
import time
import json
import gc

def retrieve_candidates_v2(s1_path, s2_path, s3_path, output_path, metrics_path=None, gt_path=None):
    print("Loading data for Retrieval V2 (Polars Optimized)...")
    t0 = time.time()
    
    cols = ["entity_id", "country_norm", "name_legal", "address_norm", "address_numbers", "name_tokens"]
    
    s1 = pl.read_parquet(s1_path, columns=cols)
    s2 = pl.read_parquet(s2_path, columns=cols)
    s3 = pl.read_parquet(s3_path, columns=cols)
    
    candidates = []
    
    def filter_freq(df, keys, max_freq):
        freq2 = s2.group_by(keys).len()
        freq3 = s3.group_by(keys).len()
        freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
        
        valid_keys = freq.filter((pl.col("len") > 0) & (pl.col("len") <= max_freq)).select(keys)
        return df.join(valid_keys, on=keys, how="inner")

    # ----------------------------------------------------
    # CH 1: Exact Name (Cap 1000)
    # ----------------------------------------------------
    print("CH 1: Exact Name (Cap 1000)")
    keys = ["name_legal", "country_norm"]
    s1_c = filter_freq(s1.filter(pl.col("name_legal") != ""), keys, 1000)
    s2_c = filter_freq(s2.filter(pl.col("name_legal") != ""), keys, 1000)
    s3_c = filter_freq(s3.filter(pl.col("name_legal") != ""), keys, 1000)
    
    c1_s2 = s1_c.join(s2_c, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("exact_name").alias("channel"))
    c1_s3 = s1_c.join(s3_c, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("exact_name").alias("channel"))
    candidates.extend([c1_s2, c1_s3])
    del s1_c, s2_c, s3_c, c1_s2, c1_s3; gc.collect()

    # ----------------------------------------------------
    # CH 2: Exact Address (Cap 1000)
    # ----------------------------------------------------
    print("CH 2: Exact Address (Cap 1000)")
    keys = ["address_norm", "country_norm"]
    s1_c = filter_freq(s1.filter(pl.col("address_norm") != ""), keys, 1000)
    s2_c = filter_freq(s2.filter(pl.col("address_norm") != ""), keys, 1000)
    s3_c = filter_freq(s3.filter(pl.col("address_norm") != ""), keys, 1000)
    
    c2_s2 = s1_c.join(s2_c, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("exact_addr").alias("channel"))
    c2_s3 = s1_c.join(s3_c, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("exact_addr").alias("channel"))
    candidates.extend([c2_s2, c2_s3])
    del s1_c, s2_c, s3_c, c2_s2, c2_s3; gc.collect()
    
    # ----------------------------------------------------
    # CH 3: Name Tokens (Cap 1000)
    # ----------------------------------------------------
    print("CH 3: Token Blocking (Cap 1000)")
    def prepare_tokens(df):
        return df.filter(pl.col("name_tokens") != "").with_columns(pl.col("name_tokens").str.split(" ")).explode("name_tokens").rename({"name_tokens": "token"}).filter(pl.col("token") != "")
        
    s1_tok = prepare_tokens(s1)
    s2_tok = prepare_tokens(s2)
    s3_tok = prepare_tokens(s3)
    
    keys = ["token", "country_norm"]
    freq2 = s2_tok.group_by(keys).len()
    freq3 = s3_tok.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_tokens = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 1000)).select(keys)
    
    s1_tok = s1_tok.join(valid_tokens, on=keys, how="inner")
    s2_tok = s2_tok.join(valid_tokens, on=keys, how="inner")
    s3_tok = s3_tok.join(valid_tokens, on=keys, how="inner")
    
    c3_s2 = s1_tok.join(s2_tok, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("token").alias("channel"))
    c3_s3 = s1_tok.join(s3_tok, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("token").alias("channel"))
    candidates.extend([c3_s2, c3_s3])
    del s1_tok, s2_tok, s3_tok, c3_s2, c3_s3, valid_tokens, freq, freq2, freq3; gc.collect()
    
    # ----------------------------------------------------
    # CH 4: Address Numbers + Name Token (Cap 100000)
    # ----------------------------------------------------
    # If they share an address number AND a single generic name token, it's a very strong signal.
    # CH 4: Address Numbers + Name Token (Cap 100000)
    # ----------------------------------------------------
    print("CH 4: Address Number + Name Token")
    s1_tok = prepare_tokens(s1).filter(pl.col("address_numbers") != "")
    s2_tok = prepare_tokens(s2).filter(pl.col("address_numbers") != "")
    s3_tok = prepare_tokens(s3).filter(pl.col("address_numbers") != "")
    
    keys = ["token", "address_numbers", "country_norm"]
    c4_s2 = s1_tok.join(s2_tok, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    c4_s3 = s1_tok.join(s3_tok, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    candidates.extend([c4_s2, c4_s3])
    del s1_tok, s2_tok, s3_tok, c4_s2, c4_s3; gc.collect()

    # ----------------------------------------------------
    # CH 5: Postal Code + Name Token (No cap)
    # ----------------------------------------------------
    print("CH 5: Postal Code + Name Token")
    # Extract postal code
    s1_postal = prepare_tokens(s1).with_columns(pl.col("address_norm").str.extract(r'\b(\d{5,6})\b').alias("postal")).filter(pl.col("postal").is_not_null())
    s2_postal = prepare_tokens(s2).with_columns(pl.col("address_norm").str.extract(r'\b(\d{5,6})\b').alias("postal")).filter(pl.col("postal").is_not_null())
    s3_postal = prepare_tokens(s3).with_columns(pl.col("address_norm").str.extract(r'\b(\d{5,6})\b').alias("postal")).filter(pl.col("postal").is_not_null())
    
    keys = ["token", "postal", "country_norm"]
    c5_s2 = s1_postal.join(s2_postal, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("tok_postal").alias("channel"))
    c5_s3 = s1_postal.join(s3_postal, on=keys, how="inner").select([pl.col("entity_id").alias("source1_entity_id"), pl.col("entity_id_right").alias("candidate_entity_id")]).with_columns(pl.lit("tok_postal").alias("channel"))
    candidates.extend([c5_s2, c5_s3])
    del s1_postal, s2_postal, s3_postal, c5_s2, c5_s3; gc.collect()


    # ----------------------------------------------------
    # UNION & EXPORT
    # ----------------------------------------------------
    print("Union and deduplicate candidates...")
    all_cands = pl.concat(candidates)
    
    all_cands = all_cands.group_by(["source1_entity_id", "candidate_entity_id"]).agg(
        pl.col("channel").unique()
    ).with_columns(
        pl.col("channel").list.join("|").alias("channels")
    ).drop("channel")
    
    t_retrieval = time.time() - t0
    print(f"Retrieval finished in {t_retrieval:.2f}s. Total candidate pairs: {all_cands.height}")
    
    all_cands.write_parquet(output_path)
    print(f"Saved candidates to {output_path}")
    
    if gt_path and metrics_path:
        gt = pl.read_csv(gt_path, separator='\t', infer_schema_length=0).fill_null("")
        gt_exploded = gt.with_columns(
            pl.col("matched_entity_ids").str.split(",")
        ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        
        total_true_links = gt_exploded.height
        
        recovered = gt_exploded.join(
            all_cands,
            left_on=["source1_entity_id", "matched_entity_ids"],
            right_on=["source1_entity_id", "candidate_entity_id"],
            how="inner"
        )
        recovered_links = recovered.height
        pair_recall = recovered_links / total_true_links if total_true_links > 0 else 0
        
        metrics = {
            "total_candidates": all_cands.height,
            "total_true_links": total_true_links,
            "recovered_links": recovered_links,
            "pair_recall": pair_recall,
            "runtime": t_retrieval
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Pair Recall: {pair_recall:.4f} ({recovered_links}/{total_true_links})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', required=True)
    parser.add_argument('--s2', required=True)
    parser.add_argument('--s3', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metrics', required=False)
    parser.add_argument('--gt', required=False)
    args = parser.parse_args()
    
    retrieve_candidates_v2(args.s1, args.s2, args.s3, args.output, args.metrics, args.gt)
