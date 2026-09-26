import polars as pl
import time
import gc
import json
import os
import glob
import duckdb
import jellyfish

def retrieval_v3(s1_path, s2_path, s3_path, output_path, gt_path=None, metrics_path=None, smoke_test=False):
    print("=" * 60)
    print("RETRIEVAL V3: Disk-Backed Safe Enhanced Multi-Channel")
    print("=" * 60)
    t0 = time.time()

    cols = ["entity_id", "country_norm", "name_legal", "name_punct",
            "name_sorted", "address_norm", "address_numbers", "name_tokens"]

    s1 = pl.read_parquet(s1_path, columns=cols)
    if smoke_test:
        print("SMOKE TEST: Using 50,000 S1 entities")
        s1 = s1.head(50000)

    s2 = pl.read_parquet(s2_path, columns=cols)
    s3 = pl.read_parquet(s3_path, columns=cols)

    print(f"  S1: {s1.height}, S2: {s2.height}, S3: {s3.height}")

    # Clear old tmp files
    for f in glob.glob(str(output_path).replace(".parquet", "_tmp_*.parquet")):
        try: os.remove(f)
        except: pass

    chunk_idx = 0

    def write_cand(df, name):
        nonlocal chunk_idx
        tmp_path = str(output_path).replace(".parquet", f"_tmp_{name}_{chunk_idx}.parquet")
        # Ensure we only write non-empty dataframes
        if df.height > 0:
            df.write_parquet(tmp_path)
        chunk_idx += 1

    def do_channel(name, s1_df, s2_df, s3_df, keys, channel_name, max_freq=None):
        s1_df = s1_df.select(keys + ["entity_id"])
        s2_df = s2_df.select(keys + ["entity_id"])
        s3_df = s3_df.select(keys + ["entity_id"])
        if max_freq is not None:
            freq2 = s2_df.group_by(keys).len()
            freq3 = s3_df.group_by(keys).len()
            freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
            valid = freq.filter((pl.col("len") > 0) & (pl.col("len") <= max_freq)).select(keys)
            s1_f = s1_df.join(valid, on=keys, how="inner")
            s2_f = s2_df.join(valid, on=keys, how="inner")
            s3_f = s3_df.join(valid, on=keys, how="inner")
        else:
            s1_f, s2_f, s3_f = s1_df, s2_df, s3_df

        c_s2 = s1_f.join(s2_f, on=keys, how="inner").select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_right").alias("candidate_entity_id")
        ]).with_columns(pl.lit(channel_name).alias("channel"))
        
        c_s3 = s1_f.join(s3_f, on=keys, how="inner").select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_right").alias("candidate_entity_id")
        ]).with_columns(pl.lit(channel_name).alias("channel"))

        total = c_s2.height + c_s3.height
        print(f"  {name}: {total:,} pairs")
        write_cand(c_s2, name)
        write_cand(c_s3, name)
        return total

    print("CH 1: Exact Legal Name")
    s1_c = s1.filter(pl.col("name_legal") != "")
    s2_c = s2.filter(pl.col("name_legal") != "")
    s3_c = s3.filter(pl.col("name_legal") != "")
    do_channel("exact_legal", s1_c, s2_c, s3_c, ["name_legal", "country_norm"], "exact_name", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    print("CH 2: Exact Punct Name")
    s1_c = s1.filter(pl.col("name_punct") != "")
    s2_c = s2.filter(pl.col("name_punct") != "")
    s3_c = s3.filter(pl.col("name_punct") != "")
    do_channel("exact_punct", s1_c, s2_c, s3_c, ["name_punct", "country_norm"], "exact_punct", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    print("CH 3: Exact Address")
    s1_c = s1.filter(pl.col("address_norm") != "")
    s2_c = s2.filter(pl.col("address_norm") != "")
    s3_c = s3.filter(pl.col("address_norm") != "")
    do_channel("exact_addr", s1_c, s2_c, s3_c, ["address_norm", "country_norm"], "exact_addr", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    print("CH 4: Sorted Name")
    s1_c = s1.filter(pl.col("name_sorted") != "")
    s2_c = s2.filter(pl.col("name_sorted") != "")
    s3_c = s3.filter(pl.col("name_sorted") != "")
    do_channel("sorted_name", s1_c, s2_c, s3_c, ["name_sorted", "country_norm"], "sorted_name", max_freq=400)
    del s1_c, s2_c, s3_c; gc.collect()

    def prepare_tokens(df):
        return (df.filter(pl.col("name_tokens") != "")
                .with_columns(pl.col("name_tokens").str.split(" "))
                .explode("name_tokens")
                .rename({"name_tokens": "token"})
                .filter(pl.col("token") != "")
                .filter(pl.col("token").str.len_chars() > 1))

    s1_tok = prepare_tokens(s1)
    s2_tok = prepare_tokens(s2)
    s3_tok = prepare_tokens(s3)

    print("CH 5: Token Blocking")
    keys = ["token", "country_norm"]
    freq2 = s2_tok.group_by(keys).len()
    freq3 = s3_tok.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_tokens = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select(keys)

    s1_tf = s1_tok.select(keys + ["entity_id"]).join(valid_tokens, on=keys, how="inner")
    s2_tf = s2_tok.select(keys + ["entity_id"]).join(valid_tokens, on=keys, how="inner")
    s3_tf = s3_tok.select(keys + ["entity_id"]).join(valid_tokens, on=keys, how="inner")

    c_s2 = s1_tf.join(s2_tf, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("token").alias("channel"))
    c_s3 = s1_tf.join(s3_tf, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("token").alias("channel"))
    print(f"  token: {c_s2.height + c_s3.height:,} pairs")
    write_cand(c_s2, "token")
    write_cand(c_s3, "token")
    del s1_tf, s2_tf, s3_tf, c_s2, c_s3, valid_tokens, freq, freq2, freq3; gc.collect()

    print("CH 6: Token + Address Number")
    s1_ta = s1_tok.filter(pl.col("address_numbers") != "")
    s2_ta = s2_tok.filter(pl.col("address_numbers") != "")
    s3_ta = s3_tok.filter(pl.col("address_numbers") != "")
    keys = ["token", "address_numbers", "country_norm"]
    
    freq2 = s2_ta.group_by(keys).len()
    freq3 = s3_ta.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_ta = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 1000)).select(keys)

    c_s2 = s1_ta.select(keys + ["entity_id"]).join(valid_ta, on=keys, how="inner").join(s2_ta.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    c_s3 = s1_ta.select(keys + ["entity_id"]).join(valid_ta, on=keys, how="inner").join(s3_ta.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    print(f"  tok_addrnum: {c_s2.height + c_s3.height:,} pairs")
    write_cand(c_s2, "tok_addrnum")
    write_cand(c_s3, "tok_addrnum")
    del s1_ta, s2_ta, s3_ta, c_s2, c_s3, valid_ta, freq, freq2, freq3; gc.collect()

    print("CH 7: Postal + Token")
    s1_postal = s1_tok.with_columns(pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")).filter(pl.col("postal").is_not_null())
    s2_postal = s2_tok.with_columns(pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")).filter(pl.col("postal").is_not_null())
    s3_postal = s3_tok.with_columns(pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")).filter(pl.col("postal").is_not_null())
    keys = ["token", "postal", "country_norm"]
    
    freq2 = s2_postal.group_by(keys).len()
    freq3 = s3_postal.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_postal = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 1000)).select(keys)

    c_s2 = s1_postal.select(keys + ["entity_id"]).join(valid_postal, on=keys, how="inner").join(s2_postal.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_postal").alias("channel"))
    c_s3 = s1_postal.select(keys + ["entity_id"]).join(valid_postal, on=keys, how="inner").join(s3_postal.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_postal").alias("channel"))
    print(f"  tok_postal: {c_s2.height + c_s3.height:,} pairs")
    write_cand(c_s2, "tok_postal")
    write_cand(c_s3, "tok_postal")
    del s1_postal, s2_postal, s3_postal, c_s2, c_s3, valid_postal, freq, freq2, freq3; gc.collect()

    print("CH 8: Token Bigram")
    def make_bigrams(tok_df):
        ent_toks = tok_df.select(["entity_id", "token", "country_norm"]).unique()
        pairs = ent_toks.join(ent_toks, on=["entity_id", "country_norm"], how="inner", suffix="_2")
        pairs = pairs.filter(pl.col("token") < pl.col("token_2"))
        pairs = pairs.with_columns((pl.col("token") + "|" + pl.col("token_2")).alias("bigram")).select(["entity_id", "country_norm", "bigram"])
        return pairs

    s1_bg = make_bigrams(s1_tok)
    s2_bg = make_bigrams(s2_tok)
    s3_bg = make_bigrams(s3_tok)

    keys = ["bigram", "country_norm"]
    freq2 = s2_bg.group_by(keys).len()
    freq3 = s3_bg.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_bg = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select(keys)

    s1_bg = s1_bg.select(keys + ["entity_id"]).join(valid_bg, on=keys, how="inner")
    s2_bg = s2_bg.select(keys + ["entity_id"]).join(valid_bg, on=keys, how="inner")
    s3_bg = s3_bg.select(keys + ["entity_id"]).join(valid_bg, on=keys, how="inner")

    c_s2 = s1_bg.join(s2_bg, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("bigram").alias("channel"))
    c_s3 = s1_bg.join(s3_bg, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("bigram").alias("channel"))
    print(f"  bigram: {c_s2.height + c_s3.height:,} pairs")
    write_cand(c_s2, "bigram")
    write_cand(c_s3, "bigram")
    del s1_bg, s2_bg, s3_bg, c_s2, c_s3, valid_bg, freq, freq2, freq3; gc.collect()
    del s1_tok, s2_tok, s3_tok; gc.collect()

    print("CH 9: Phonetic")
    def add_soundex(df):
        names = df["name_legal"].to_list()
        soundex_codes = []
        for n in names:
            if n and n.strip():
                first_tok = n.split()[0] if n.split() else ""
                if first_tok and first_tok.isalpha():
                    try:
                        soundex_codes.append(jellyfish.soundex(first_tok))
                    except:
                        soundex_codes.append("")
                else:
                    soundex_codes.append("")
            else:
                soundex_codes.append("")
        return df.with_columns(pl.Series("soundex", soundex_codes))

    s1_sx = add_soundex(s1).filter(pl.col("soundex") != "")
    s2_sx = add_soundex(s2).filter(pl.col("soundex") != "")
    s3_sx = add_soundex(s3).filter(pl.col("soundex") != "")

    keys = ["soundex", "country_norm"]
    freq2 = s2_sx.group_by(keys).len()
    freq3 = s3_sx.group_by(keys).len()
    freq = pl.concat([freq2, freq3]).group_by(keys).agg(pl.col("len").sum())
    valid_sx = freq.filter((pl.col("len") > 0) & (pl.col("len") <= 400)).select(keys)

    s1_sx = s1_sx.select(keys + ["entity_id"]).join(valid_sx, on=keys, how="inner")
    s2_sx = s2_sx.select(keys + ["entity_id"]).join(valid_sx, on=keys, how="inner")
    s3_sx = s3_sx.select(keys + ["entity_id"]).join(valid_sx, on=keys, how="inner")

    c_s2 = s1_sx.join(s2_sx, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("phonetic").alias("channel"))
    c_s3 = s1_sx.join(s3_sx, on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("phonetic").alias("channel"))
    print(f"  phonetic: {c_s2.height + c_s3.height:,} pairs")
    write_cand(c_s2, "phonetic")
    write_cand(c_s3, "phonetic")
    del s1_sx, s2_sx, s3_sx, c_s2, c_s3; gc.collect()

    print("Union and deduplicate via DuckDB streaming...")
    tmp_files_glob = str(output_path).replace(".parquet", "_tmp_*.parquet")
    con = duckdb.connect()
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    
    query = f"""
    COPY (
        SELECT 
            source1_entity_id, 
            candidate_entity_id,
            string_agg(DISTINCT channel, '|') as channels,
            count(DISTINCT channel) as num_channels
        FROM read_parquet('{tmp_files_glob}')
        GROUP BY 1, 2
    ) TO '{output_path}' (FORMAT 'parquet')
    """
    con.execute(query)

    for f in glob.glob(tmp_files_glob):
        try: os.remove(f)
        except: pass

    # Compute metrics using DuckDB without loading everything into memory
    print("Computing metrics via DuckDB...")
    total_candidates = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    
    s1_stats = con.execute(f"""
        SELECT 
            avg(cnt), 
            quantile_cont(cnt, 0.5), 
            quantile_cont(cnt, 0.95), 
            quantile_cont(cnt, 0.99), 
            max(cnt)
        FROM (
            SELECT source1_entity_id, count(*) as cnt 
            FROM read_parquet('{output_path}') 
            GROUP BY 1
        )
    """).fetchone()

    t_ret = time.time() - t0
    print(f"Retrieval V3 finished in {t_ret:.1f}s. Total candidate pairs: {total_candidates:,}")

    if gt_path and metrics_path:
        gt = pl.read_csv(gt_path, separator='\t', infer_schema_length=0).fill_null("")
        gt_exploded = gt.with_columns(
            pl.col("matched_entity_ids").str.split(",")
        ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        
        # Write GT to a temporary parquet to join in DuckDB
        tmp_gt = str(output_path).replace(".parquet", "_tmp_gt.parquet")
        gt_exploded.write_parquet(tmp_gt)
        
        total_true = gt_exploded.height
        
        recovered_links = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT source1_entity_id, matched_entity_ids as candidate_entity_id FROM read_parquet('{tmp_gt}')
            ) gt
            JOIN read_parquet('{output_path}') c
            ON gt.source1_entity_id = c.source1_entity_id AND gt.candidate_entity_id = c.candidate_entity_id
        """).fetchone()[0]
        
        pair_recall = recovered_links / total_true if total_true > 0 else 0
        
        s1_with_matches = gt.filter(pl.col("matched_entity_ids") != "").height
        
        s1_with_any_recovered = con.execute(f"""
            SELECT COUNT(DISTINCT gt.source1_entity_id) FROM (
                SELECT source1_entity_id, matched_entity_ids as candidate_entity_id FROM read_parquet('{tmp_gt}')
            ) gt
            JOIN read_parquet('{output_path}') c
            ON gt.source1_entity_id = c.source1_entity_id AND gt.candidate_entity_id = c.candidate_entity_id
        """).fetchone()[0]
        
        entity_recall = s1_with_any_recovered / s1_with_matches if s1_with_matches > 0 else 0
        
        metrics = {
            "total_candidates": total_candidates,
            "total_true_links": total_true,
            "recovered_links": recovered_links,
            "pair_recall": pair_recall,
            "entity_recall": entity_recall,
            "runtime": t_ret,
            "mean_candidates_per_s1": float(s1_stats[0]),
            "median_candidates": float(s1_stats[1]),
            "p95_candidates": float(s1_stats[2]),
            "p99_candidates": float(s1_stats[3]),
            "max_candidates": float(s1_stats[4]),
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"  Pair Recall: {pair_recall:.4f} ({recovered_links}/{total_true})")
        print(f"  Entity Recall (at-least-one): {entity_recall:.4f}")
        try: os.remove(tmp_gt)
        except: pass

    con.close()
    return t_ret
