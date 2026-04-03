"""
Normalize the branding palette dataset into a retrieval-ready table.

Expected output:
data/processed/branding_palettes_processed.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


RAW_PATH = os.environ.get(
    "BRANDING_PALETTE_RAW_CSV",
    "data/raw/emotion_palette.csv"
)
OUT_PATH = os.environ.get(
    "BRAND_PALETTE_CSV",
    "data/processed/branding_palettes_processed.csv"
)


def _normalize(c: str) -> str:
    return str(c).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def main() -> None:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing raw palette file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df.columns = [_normalize(c) for c in df.columns]

    # 兼容 Color 1 / color 1 / color_1 这种列名
    rename_candidates = {
        "color_1": "color1",
        "color_2": "color2",
        "color_3": "color3",
        "color_4": "color4",
        "color_5": "color5",
    }
    df = df.rename(columns=rename_candidates)

    required_color_cols = ["color1", "color2", "color3", "color4", "color5"]
    missing = [c for c in required_color_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected color columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    keep_cols = required_color_cols.copy()

    # 保留可能的元数据列
    for meta_col in ["palette_name", "name", "industry", "brand", "id", "palette_id"]:
        if meta_col in df.columns and meta_col not in keep_cols:
            keep_cols.append(meta_col)

    # 保留 0/1 标签列
    for c in df.columns:
        if c in keep_cols:
            continue
        vals = set(df[c].dropna().astype(str).str.lower().unique().tolist())
        if vals and vals.issubset({"0", "1", "0.0", "1.0", "true", "false"}):
            keep_cols.append(c)

    out = df[keep_cols].copy()

    if "name" in out.columns and "palette_name" not in out.columns:
        out = out.rename(columns={"name": "palette_name"})

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved processed palette table to {OUT_PATH}")
    print(f"Output columns: {list(out.columns)}")
    print(f"Number of rows: {len(out)}")


if __name__ == "__main__":
    main()
