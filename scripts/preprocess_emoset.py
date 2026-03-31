"""
Preprocess EmoSet annotations into a compact emotion summary table.

Expected output:
data/processed/emoset_emotion_summary.csv

Input CSV should contain columns roughly like:
- emotion
- brightness
- colorfulness

If raw column names differ, the script will try to infer them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


RAW_PATH = os.environ.get("EMOSET_RAW_CSV", "data/raw/emoset_annotations.csv")
OUT_PATH = os.environ.get(
    "EMOSET_SUMMARY_PATH",
    "data/processed/emoset_emotion_summary.csv"
)


def _normalize(c: str) -> str:
    return str(c).strip().lower().replace("-", "_").replace(" ", "_")


def _find_col(cols, candidates):
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def main() -> None:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing EmoSet raw file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df.columns = [_normalize(c) for c in df.columns]

    emotion_col = _find_col(df.columns, ["emotion", "emotion_label", "category"])
    brightness_col = _find_col(df.columns, ["brightness", "brightness_mean"])
    colorfulness_col = _find_col(df.columns, ["colorfulness", "colorfulness_mean"])

    if not all([emotion_col, brightness_col, colorfulness_col]):
        raise ValueError(
            f"Could not infer required columns from {list(df.columns)}"
        )

    out = (
        df[[emotion_col, brightness_col, colorfulness_col]]
        .rename(columns={
            emotion_col: "emotion",
            brightness_col: "brightness",
            colorfulness_col: "colorfulness",
        })
        .dropna()
        .groupby("emotion", as_index=False)
        .agg(
            brightness_mean=("brightness", "mean"),
            colorfulness_mean=("colorfulness", "mean"),
            sample_count=("emotion", "size"),
        )
        .sort_values("sample_count", ascending=False)
    )

    Path(os.path.dirname(OUT_PATH)).mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved EmoSet summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
