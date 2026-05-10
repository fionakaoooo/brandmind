# Data Preprocessing

## Google Fonts
- Extracted: family name, category, variants, subsets
- Filtered by archetype-appropriate categories
- Ranked by popularity via Google Fonts API

## Kaggle Emotion Palettes
- 745 palettes with binary emotion labels
- Scored by emotion match count
- Re-ranked using EmoSet brightness/colorfulness targets
- Anchored with #FFFFFF and #1A1A1A for WCAG compliance

## EmoSet
- Used for brightness_target and colorfulness_target per emotion
- Provides visual grounding for palette re-ranking

## Design Heuristics
- Curated from design literature
- Stored as JSON rule bank
- Retrieved by brand attribute keywords
- Weights updated after each QC evaluation
