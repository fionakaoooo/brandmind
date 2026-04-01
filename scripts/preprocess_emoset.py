import json
import os
from collections import defaultdict
import pandas as pd

RAW_DIR = "data/raw/emo_set"
OUT_PATH = "data/processed/emoset_emotion_summary.csv"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    files = ["train.json", "val.json", "test.json"]

    brightness_by_emotion = defaultdict(list)
    colorfulness_by_emotion = defaultdict(list)

    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        data = load_json(path)

        for item in data:
            emotion = item.get("emotion")
            brightness = item.get("brightness")
            colorfulness = item.get("colorfulness")

            if emotion is None:
                continue

            if brightness is not None:
                brightness_by_emotion[emotion].append(float(brightness))

            if colorfulness is not None:
                colorfulness_by_emotion[emotion].append(float(colorfulness))

    all_emotions = sorted(set(brightness_by_emotion) | set(colorfulness_by_emotion))

    rows = []
    for emo in all_emotions:
        bvals = brightness_by_emotion.get(emo, [])
        cvals = colorfulness_by_emotion.get(emo, [])

        rows.append({
            "emotion": emo,
            "brightness_mean": sum(bvals) / len(bvals) if bvals else 0.5,
            "colorfulness_mean": sum(cvals) / len(cvals) if cvals else 0.5,
            "count_brightness": len(bvals),
            "count_colorfulness": len(cvals),
        })

    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved to {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
