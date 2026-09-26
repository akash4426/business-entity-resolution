import polars as pl
import argparse
from pathlib import Path

def generate_decisions(predictions_path, s1_path, output_path, candidate_out_path, threshold=0.5):
    print(f"Loading predictions from {predictions_path}...")
    
    # We must ensure all S1 entities are present. 
    # Read the full S1 split to ensure zero-candidate S1s are output correctly (as empty)
    s1 = pl.read_parquet(s1_path)
    all_s1 = s1.select(["entity_id"]).rename({"entity_id": "source1_entity_id"})
    
    # ----------------------------------------------------
    # Candidate Set Export (REQUIRED OUTPUT 2)
    # ----------------------------------------------------
    # "Every final predicted match must exist in the candidate set."
    # The candidate pairs file should have format: source1_entity_id \t candidate_entity_ids
    print("Generating candidate_pairs.tsv...")
    cands_grouped = pl.scan_parquet(predictions_path).select([
        "source1_entity_id", "candidate_entity_id"
    ]).group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("candidates")
    ).with_columns(
        pl.col("candidates").list.join(",")
    ).collect(streaming=True)
    all_s1 = s1.select(["entity_id"]).rename({"entity_id": "source1_entity_id"})
    
    final_cands = all_s1.join(cands_grouped, on="source1_entity_id", how="left").fill_null("")
    
    final_cands.select([
        "source1_entity_id", 
        pl.col("candidates").alias("candidate_entity_ids")
    ]).write_csv(candidate_out_path, separator='\t', quote_style='never')
    
    # ----------------------------------------------------
    # Final Matches Export (REQUIRED OUTPUT 1)
    # ----------------------------------------------------
    print("Generating matching_results.tsv...")
    matches_grouped = pl.scan_parquet(predictions_path).filter(
        pl.col("probability") >= threshold
    ).select([
        "source1_entity_id", "candidate_entity_id"
    ]).group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("matches")
    ).with_columns(
        pl.col("matches").list.join(",")
    ).collect(streaming=True)
    
    final_matches = all_s1.join(matches_grouped, on="source1_entity_id", how="left").fill_null("")
    
    final_matches.select([
        "source1_entity_id",
        pl.col("matches").alias("matched_entity_ids")
    ]).write_csv(output_path, separator='\t', quote_style='never')
    
    print("Decision engine complete.")
    
    # Print some stats
    zero_matches = final_matches.filter(pl.col("matches") == "").height
    singletons = final_matches.filter(pl.col("matches").str.contains(",") == False).filter(pl.col("matches") != "").height
    multi = final_matches.height - zero_matches - singletons
    
    print(f"Total S1: {final_matches.height}")
    print(f"Zero matches: {zero_matches}")
    print(f"Single matches: {singletons}")
    print(f"Multi matches: {multi}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--s1', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate_out', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    generate_decisions(args.predictions, args.s1, args.output, args.candidate_out, args.threshold)
