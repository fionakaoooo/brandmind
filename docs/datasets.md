# Datasets Used

## 1. Google Fonts
- Source: fonts.google.com API
- Content: 1,500+ font families with metadata
- Used by: font_lookup() in Agent 2
- Purpose: Grounded font recommendations

## 2. Emotion-Labeled Color Palettes (Kaggle)
- Source: kaggle.com/datasets/programmers3/emotion-labeled-color-palettes-for-branding
- Content: 745 five-color palettes with emotion labels
- Used by: color_retrieve() in Agent 2
- Purpose: Emotion-aware palette retrieval

## 3. EmoSet
- Source: vcc.tech/EmoSet
- Content: Visual emotion dataset with brightness/colorfulness targets
- Used by: color_retrieve() re-ranking
- Purpose: Align palette brightness/colorfulness with archetype

## 4. Curated Design Heuristics
- Source: Design literature + Ellen Lupton Thinking with Type
- Content: 50-100 design rules per archetype
- Used by: heuristic_search() in Agent 2
- Purpose: Ground design decisions in established principles
