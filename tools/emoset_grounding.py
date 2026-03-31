from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import pandas as pd


EMOSET_SUMMARY_PATH = os.environ.get(
    "EMOSET_SUMMARY_PATH",
    "data/processed/emoset_emotion_summary.csv"
)


def _normalize(text: str) -> str:
    return (
        str(text).strip().lower().replace("-", "_").replace(" ", "_")
    )


@lru_cache(maxsize=1)
def load_emoset_summary() -> pd.DataFrame:
    if not os.path.exists(EMOSET_SUMMARY_PATH):
        raise FileNotFoundError(
            f"Missing EmoSet summary file at {EMOSET_SUMMARY_PATH}"
        )

    df = pd.read_csv(EMOSET_SUMMARY_PATH)
    df.columns = [_normalize(c) for c in df.columns]
    return df


def emotion_to_visual_profile(emotions: List[str]) -> Dict[str, float]:
    df = load_emoset_summary()
    wanted = {_normalize(e) for e in emotions}

    hit = df[df["emotion"].astype(str).map(_normalize).isin(wanted)]
    if hit.empty:
        return {
            "brightness_target": 0.55,
            "colorfulness_target": 0.55,
        }

    return {
        "brightness_target": float(hit["brightness_mean"].mean()),
        "colorfulness_target": float(hit["colorfulness_mean"].mean()),
    }
