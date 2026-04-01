# 1. First, execute the extraction command in Colab
import os
print("📦 Extracting EmoSet.zip (this might take a minute or two)...")
os.system("unzip -q EmoSet.zip -d /content/extracted_data")
print("✅ Extraction complete!")

# 2. Import necessary libraries
import json
import pandas as pd
from collections import defaultdict

# 3. Smart search for the actual path of the annotation folder
def find_annotation_folder(start_path):
    for root, dirs, files in os.walk(start_path):
        if 'annotation' in dirs:
            return os.path.join(root, 'annotation')
    return None

def process_emoset_to_csv():
    print("🔍 Searching for the annotation data folder...")
    annotation_folder = find_annotation_folder("/content/extracted_data")
    
    if not annotation_folder:
        print("❌ Error: Could not find the 'annotation' folder in the zip file. Please check your EmoSet.zip!")
        return
        
    print(f"🎯 Found it! Path: {annotation_folder}")
    print("⏳ Starting to scan ~118,000 JSON files to extract academic features...")

    emotion_stats = defaultdict(lambda: {"brightness_sum": 0.0, "colorfulness_sum": 0.0, "count": 0})
    processed_count = 0

    # Recursively traverse all JSON files
    for root, _, files in os.walk(annotation_folder):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        emotion = data.get("emotion")
                        brightness = data.get("brightness")
                        colorfulness = data.get("colorfulness")
                        
                        if emotion and brightness is not None and colorfulness is not None:
                            emotion = emotion.strip().lower()
                            emotion_stats[emotion]["brightness_sum"] += float(brightness)
                            emotion_stats[emotion]["colorfulness_sum"] += float(colorfulness)
                            emotion_stats[emotion]["count"] += 1
                            
                    processed_count += 1
                    if processed_count % 10000 == 0:
                        print(f"   Processed {processed_count} files...")
                except Exception:
                    pass
                    
    print(f"\n✅ Scan complete! Extracted data from {processed_count} files.")
    print("⚙️ Generating CSV file...")

    # Calculate means and generate the dataframe
    records = []
    for emo, stats in emotion_stats.items():
        if stats["count"] > 0:
            records.append({
                "emotion": emo,
                "brightness_mean": round(stats["brightness_sum"] / stats["count"], 4),
                "colorfulness_mean": round(stats["colorfulness_sum"] / stats["count"], 4)
            })

    df = pd.DataFrame(records)
    df.sort_values(by="emotion", inplace=True)
    
    # ====== Core path configuration ======
    # Colab will generate this final file in the left panel
    output_csv_path = "/content/emoset_emotion_summary.csv"
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    # =====================================
    
    print(f"🎉 Success! Feature summary file saved to: {output_csv_path}")
    print("\n📊 Data Preview:")
    print(df.to_string(index=False))
    print("\n👉 Now you can click the folder icon on the left side of Colab 📁")
    print("👉 Find 'emoset_emotion_summary.csv', right-click and select Download!")

# 4. Execute the main program
if __name__ == "__main__":
    process_emoset_to_csv()
