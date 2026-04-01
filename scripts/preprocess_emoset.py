import json
import os
import pandas as pd
from collections import defaultdict
import time

def process_emoset_on_mac(dataset_path: str, output_csv: str):
    """
    Optimizing EmoSet preprocessing for macOS Desktop environments.
    """
    # Converting tilde to full home directory path
    full_path = os.path.expanduser(dataset_path)
    
    # Locating the required annotation sub-folder
    annotation_folder = os.path.join(full_path, "annotation")
    
    if not os.path.exists(annotation_folder):
        print(f"Error: Annotation folder not found at {full_path}")
        print("Suggestion: Verifying if the folder name is 'EmoSet' and containing 'annotation' directory.")
        return

    print(f"Initializing processing for dataset at: {annotation_folder}")
    print("Reading approximately 118,000 JSON files (estimated time: 1-2 minutes)...")

    emotion_stats = defaultdict(lambda: {"b_sum": 0.0, "c_sum": 0.0, "count": 0})
    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Iterating through all sub-directories (e.g., amusement, awe, etc.)
    for root, _, files in os.walk(annotation_folder):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extracting core academic features
                        emo = data.get("emotion")
                        b = data.get("brightness")
                        c = data.get("colorfulness")
                        
                        if emo and b is not None and c is not None:
                            emo = str(emo).strip().lower()
                            emotion_stats[emo]["b_sum"] += float(b)
                            emotion_stats[emo]["c_sum"] += float(c)
                            emotion_stats[emo]["count"] += 1
                            
                    processed_count += 1
                    # Logging progress every 10,000 files
                    if processed_count % 10000 == 0:
                        elapsed = time.time() - start_time
                        print(f"Processing {processed_count} files... (Elapsed: {elapsed:.1f}s)")
                except Exception:
                    error_count += 1
                    continue

    if processed_count == 0:
        print("Error: No valid data extracted. Checking if JSON contents follow EmoSet standards.")
        return

    print(f"\nScanning completed. Success: {processed_count}, Failed: {error_count}")
    print("Calculating mean values for brightness and colorfulness per emotion...")

    # Aggregating final data
    final_data = []
    for emotion_name, stats in emotion_stats.items():
        if stats["count"] > 0:
            final_data.append({
                "emotion": emotion_name,
                "brightness_mean": round(stats["b_sum"] / stats["count"], 4),
                "colorfulness_mean": round(stats["c_sum"] / stats["count"], 4),
                "sample_size": stats["count"]
            })

    # Converting to DataFrame and sorting alphabetically
    df = pd.DataFrame(final_data).sort_values(by="emotion")
    
    # Saving results to the current script directory
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    total_time = time.time() - start_time
    print(f"Task finished. Total duration: {total_time:.1f} seconds.")
    print(f"Resulting file saved at: {os.path.abspath(output_csv)}")
    print("\n--- Data Preview (Top 10 Emotions) ---")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    # ======================================================
    # Path Configuration
    # ======================================================
    
    # Using ~/Desktop to automatically target current Mac user's desktop
    DATASET_LOCATION = "~/Desktop/EmoSet-118k" 
    
    # Defining output filename
    RESULT_NAME = "emoset_emotion_summary.csv"
    
    process_emoset_on_mac(DATASET_LOCATION, RESULT_NAME)
