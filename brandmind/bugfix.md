# Bug Fix Record

## 2026-04-02: Google Fonts key env-name mismatch

- **Modified file:** `tools/font_lookup.py`
- **Modified location:** API key loading logic used by `fetch_all_google_fonts()`
- **Reason:**
  - `README.md` asks users to configure `GOOGLE_FONTS_KEY`.
  - Existing code only read `GOOGLE_FONTS_API_KEY`.
  - As a result, users who followed `README.md` still hit mock fallback.

## Repro command before fix

Run this command in project root:

```bash
python -c "import os; os.environ.pop('GOOGLE_FONTS_API_KEY', None); os.environ['GOOGLE_FONTS_KEY']='demo-key'; from tools.font_lookup import fetch_all_google_fonts; fetch_all_google_fonts()"
```

Expected buggy behavior before fix:

- Prints:
  - `Warning: GOOGLE_FONTS_API_KEY is not set. Using a fallback mock list.`

## Fix summary

- Added dual-key support:
  - `GOOGLE_FONTS_API_KEY` (existing)
  - `GOOGLE_FONTS_KEY` (README-compatible alias)
- Key is now resolved at call time, reducing import-time env binding issues.
