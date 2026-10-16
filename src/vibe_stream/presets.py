"""The instructions the released students were trained on, and the clip sets used for training and evaluation."""

PROMPTS = {
    "oil": "transfer this to oil painting style, vibrant colors",
    "anime": "transfer this to anime art style, clean cel shading",
}

# (source, target) captions for the CLIP scores of scripts/eval_student.py, per prompt key
CLIP_CAPTIONS = {
    "oil": ("a photo", "an oil painting with vibrant colors"),
    "anime": ("a photo", "an anime illustration with clean cel shading"),
}

# Training windows come from DAVIS-2017 sequences that dreamlite-stream does not evaluate on
# (assets/davis_heldout20_mp4); evaluation uses four of its 10 tracked clips (assets/davis_mp4).
TRAIN_CLIPS_DIR = "assets/davis_heldout20_mp4"
EVAL_CLIPS_DIR = "assets/davis_mp4"
EVAL_CLIPS = ("parkour", "libby", "blackswan", "dance-twirl")
