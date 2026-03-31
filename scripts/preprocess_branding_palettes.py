"""
Normalize the Kaggle branding palette dataset into a retrieval-ready table.

Expected output:
data/processed/branding_palettes_processed.csv
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


RAW_PATH = os.environ.get(
    "BRANDING_PALETTE_RAW_CSV",
    "data/raw/emotion_labeled_color_palettes.csv"
)
OUT_PATH = os.environ.get(
    "BRAND_PALETTE_CSV",
    "data/processed/branding_palettes_processed.csv"
)

HEX_RE = re.compile(r"^#(?:[0-9A-Fa-f]{6})$")


def _normalize(c: str) -> str:
    return str(c).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def _is_hex_column(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).head(30).tolist()
    if not vals:
        return False
    return sum(bool(HEX_RE.match(v.strip())) for v in vals) >= max(5, len(vals) // 2)


def main() -> None:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing raw palette file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df.columns = [_normalize(c) for c in df.columns]

    hex_cols = [c for c in df.columns if _is_hex_column(df[c])]
    if len(hex_cols) < 5:
        common = [c for c in ["color1", "color2", "color3", "color4", "color5"] if c in df.columns]
        hex_cols = common

    if len(hex_cols) < 5:
        raise ValueError(f"Could not identify 5 hex columns from {list(df.columns)}")

    keep_cols = list(hex_cols[:5])

    # preserve likely metadata
    for meta_col in ["palette_name", "name", "industry", "brand", "id", "palette_id"]:
        if meta_col in df.columns and meta_col not in keep_cols:
            keep_cols.append(meta_col)

    # keep binary-ish label columns
    for c in df.columns:
        if c in keep_cols:
            continue
        vals = set(df[c].dropna().astype(str).str.lower().unique().tolist())
        if vals and vals.issubset({"0", "1", "0.0", "1.0", "true", "false"}):
            keep_cols.append(c)

    out = df[keep_cols].copy()

    # rename first 5 hex columns into stable names
    rename_map = {old: f"color{i+1}" for i, old in enumerate(hex_cols[:5])}
    out = out.rename(columns=rename_map)

    if "name" in out.columns and "palette_name" not in out.columns:
        out = out.rename(columns={"name": "palette_name"})

    Path(os.path.dirname(OUT_PATH)).mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved processed palette table to {OUT_PATH}")


if __name__ == "__main__":
    main()
