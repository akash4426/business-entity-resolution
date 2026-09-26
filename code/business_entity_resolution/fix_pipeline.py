import re

file_path = "/Users/akashmacbook/Desktop/business-entity-resolution/code/business_entity_resolution/run_final_pipeline.py"

with open(file_path, 'r') as f:
    content = f.read()

# Replace uncapped channel logic with capped logic
# For CH 6
ch6_old = """    # CH 6: Name Token + Address Number intersection (uncapped)
    print("CH 6: Token + Address Number")
    s1_ta = s1_tok.filter(pl.col("address_numbers") != "")
    s2_ta = s2_tok.filter(pl.col("address_numbers") != "")
    s3_ta = s3_tok.filter(pl.col("address_numbers") != "")
    keys = ["token", "address_numbers", "country_norm"]
    c_s2 = s1_ta.select(keys + ["entity_id"]).join(s2_ta.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    c_s3 = s1_ta.select(keys + ["entity_id"]).join(s3_ta.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_addrnum").alias("channel"))
    print(f"  tok_addrnum: {c_s2.height + c_s3.height:,} pairs")
    candidates.extend([c_s2, c_s3])
    del s1_ta, s2_ta, s3_ta, c_s2, c_s3; gc.collect()"""

ch6_new = """    # CH 6: Name Token + Address Number intersection (capped)
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
    candidates.extend([c_s2, c_s3])
    del s1_ta, s2_ta, s3_ta, c_s2, c_s3, valid_ta, freq, freq2, freq3; gc.collect()"""

# For CH 7
ch7_old = """    # CH 7: Postal Code + Name Token (uncapped)
    print("CH 7: Postal + Token")
    s1_postal = s1_tok.with_columns(
        pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    s2_postal = s2_tok.with_columns(
        pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    s3_postal = s3_tok.with_columns(
        pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    keys = ["token", "postal", "country_norm"]
    c_s2 = s1_postal.select(keys + ["entity_id"]).join(s2_postal.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_postal").alias("channel"))
    c_s3 = s1_postal.select(keys + ["entity_id"]).join(s3_postal.select(keys + ["entity_id"]), on=keys, how="inner").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).with_columns(pl.lit("tok_postal").alias("channel"))
    print(f"  tok_postal: {c_s2.height + c_s3.height:,} pairs")
    candidates.extend([c_s2, c_s3])
    del s1_postal, s2_postal, s3_postal, c_s2, c_s3; gc.collect()"""

ch7_new = """    # CH 7: Postal Code + Name Token (capped)
    print("CH 7: Postal + Token")
    s1_postal = s1_tok.with_columns(
        pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    s2_postal = s2_tok.with_columns(
        pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
    s3_postal = s3_tok.with_columns(
        pl.col("address_norm").str.extract(r'\\b(\\d{5,6})\\b').alias("postal")
    ).filter(pl.col("postal").is_not_null())
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
    candidates.extend([c_s2, c_s3])
    del s1_postal, s2_postal, s3_postal, c_s2, c_s3, valid_postal, freq, freq2, freq3; gc.collect()"""

# We also change how candidates are aggregated to avoid out of memory during concat
concat_old = """    # UNION & DEDUPLICATE
    print("Union and deduplicate...")
    all_cands = pl.concat(candidates)
    del candidates; gc.collect()

    all_cands = all_cands.group_by(["source1_entity_id", "candidate_entity_id"]).agg(
        pl.col("channel").unique()
    ).with_columns(
        pl.col("channel").list.join("|").alias("channels"),
        pl.col("channel").list.len().cast(pl.UInt8).alias("num_channels")
    ).drop("channel")"""

concat_new = """    # UNION & DEDUPLICATE
    print("Union and deduplicate...")
    # Write to a temporary parquet to save memory, then read it back
    pl.concat(candidates).write_parquet(str(output_path) + "_tmp.parquet")
    del candidates
    gc.collect()

    all_cands = pl.scan_parquet(str(output_path) + "_tmp.parquet").group_by(["source1_entity_id", "candidate_entity_id"]).agg(
        pl.col("channel").unique()
    ).with_columns(
        pl.col("channel").list.join("|").alias("channels"),
        pl.col("channel").list.len().cast(pl.UInt8).alias("num_channels")
    ).drop("channel").collect(engine="streaming")
    
    import os
    try:
        os.remove(str(output_path) + "_tmp.parquet")
    except:
        pass"""

content = content.replace(ch6_old, ch6_new)
content = content.replace(ch7_old, ch7_new)
content = content.replace(concat_old, concat_new)

with open(file_path, 'w') as f:
    f.write(content)
print("Updated run_final_pipeline.py successfully!")
