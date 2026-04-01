EMOTION_PRIORS = {
    "amusement": {"brightness_target": 0.8, "colorfulness_target": 0.8},
    "awe": {"brightness_target": 0.6, "colorfulness_target": 0.7},
    "contentment": {"brightness_target": 0.7, "colorfulness_target": 0.6},
    "excitement": {"brightness_target": 0.6, "colorfulness_target": 0.5},
    "anger": {"brightness_target": 0.5, "colorfulness_target": 0.7},
    "disgust": {"brightness_target": 0.7, "colorfulness_target": 0.7},
    "fear": {"brightness_target": 0.3, "colorfulness_target": 0.6},
    "sadness": {"brightness_target": 0.4, "colorfulness_target": 0.7},
    # brand-style fallback
    "calm": {"brightness_target": 0.65, "colorfulness_target": 0.35},
    "elegant": {"brightness_target": 0.6, "colorfulness_target": 0.3},
    "modern": {"brightness_target": 0.6, "colorfulness_target": 0.4},
    "clean": {"brightness_target": 0.75, "colorfulness_target": 0.25},
    "playful": {"brightness_target": 0.75, "colorfulness_target": 0.75},
    "friendly": {"brightness_target": 0.72, "colorfulness_target": 0.6},
    "bold": {"brightness_target": 0.55, "colorfulness_target": 0.8},
    "energetic": {"brightness_target": 0.65, "colorfulness_target": 0.8},
    "organic": {"brightness_target": 0.6, "colorfulness_target": 0.45},
    "fresh": {"brightness_target": 0.75, "colorfulness_target": 0.55},
    "classic": {"brightness_target": 0.5, "colorfulness_target": 0.3},
    "grounded": {"brightness_target": 0.45, "colorfulness_target": 0.35},
}

def emotion_to_visual_profile(emotions):
    vals = [EMOTION_PRIORS[e.lower()] for e in emotions if e.lower() in EMOTION_PRIORS]
    if not vals:
        return {"brightness_target": 0.55, "colorfulness_target": 0.55}
    return {
        "brightness_target": sum(v["brightness_target"] for v in vals) / len(vals),
        "colorfulness_target": sum(v["colorfulness_target"] for v in vals) / len(vals),
    }
