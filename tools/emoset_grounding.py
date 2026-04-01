import pandas as pd

EMOSET_PATH = "data/processed/emoset_emotion_summary.csv"


def emotion_to_visual_profile(emotions):
    df = pd.read_csv(EMOSET_PATH)
    hit = df[df["emotion"].isin(emotions)]

    if hit.empty:
        return {
            "brightness_target": 0.55,
            "colorfulness_target": 0.55
        }

    return {
        "brightness_target": float(hit["brightness_mean"].mean()),
        "colorfulness_target": float(hit["colorfulness_mean"].mean())
    }
