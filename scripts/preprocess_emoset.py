import json
import os
from collections import defaultdict
import pandas as pd

RAW_DIR = "data/raw/emo_set"
OUT_PATH = "data/processed/emoset_emotion_summary.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scalar(d, candidate_keys):
    """
    从 annotation dict 里尽量鲁棒地取字段
    """
    for key in candidate_keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def main():
    split_files = ["train.json", "val.json", "test.json"]

    brightness_by_emotion = defaultdict(list)
    colorfulness_by_emotion = defaultdict(list)

    for split_name in split_files:
        split_path = os.path.join(RAW_DIR, split_name)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")

        records = load_json(split_path)
        print(f"{split_name}: loaded {len(records)} index records")

        for rec in records:
            # 你的格式看起来像:
            # ["amusement", "image/amusement/xxx.jpg", "annotation/amusement/xxx.json"]
            if not isinstance(rec, list) or len(rec) < 3:
                continue

            emotion = str(rec[0]).strip().lower()
            annotation_rel_path = rec[2]

            annotation_path = os.path.join(RAW_DIR, annotation_rel_path)
            if not os.path.exists(annotation_path):
                # 有些数据可能路径相对根目录而不是 RAW_DIR
                alt_path = os.path.join("data/raw/emoset", annotation_rel_path)
                if os.path.exists(alt_path):
                    annotation_path = alt_path
                else:
                    continue

            ann = load_json(annotation_path)

            # 根据 EmoSet 常见字段尝试提取
            brightness = extract_scalar(ann, ["brightness"])
            colorfulness = extract_scalar(ann, ["colorfulness"])

            try:
                if brightness is not None:
                    brightness_by_emotion[emotion].append(float(brightness))
            except (TypeError, ValueError):
                pass

            try:
                if colorfulness is not None:
                    colorfulness_by_emotion[emotion].append(float(colorfulness))
            except (TypeError, ValueError):
                pass

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
    print(df)


if __name__ == "__main__":
    main()
