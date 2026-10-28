"""The instructions the released students were trained on, and the clip sets used for training and evaluation."""

PROMPTS = {
    "oil": "transfer this to oil painting style, vibrant colors",
    "anime": "transfer this to anime art style, clean cel shading",
}

# More instructions, for students trained on many (make_teacher_data.py --prompts) and for evaluation. The last four
# are kept out of training to test instructions a student has not seen.
MORE_PROMPTS = {
    "watercolor": "transfer this to watercolor painting style, soft washes of color",
    "pencil": "transfer this to a pencil sketch, graphite shading on paper",
    "ukiyoe": "transfer this to ukiyo-e woodblock print style, flat colors and bold outlines",
    "impressionist": "transfer this to impressionist painting style, visible dabs of paint and light",
    "comic": "transfer this to comic book style, bold ink lines and halftone shading",
    "charcoal": "transfer this to a charcoal drawing, smudged black strokes",
    "pastel": "transfer this to soft pastel drawing style, chalky texture",
    "stainedglass": "transfer this to stained glass style, colored glass pieces with dark leading",
    "popart": "transfer this to pop art style, saturated flat colors and dots",
    "winter": "make it winter, covered in snow",
    "night": "make it night, lit by street lamps",
    "sunset": "make it sunset, warm golden light",
    # held out
    "gouache": "transfer this to gouache painting style, matte opaque paint",
    "crayon": "transfer this to a crayon drawing, waxy colored strokes",
    "autumn": "make it autumn, orange and red leaves",
    "fog": "make it a foggy morning, soft mist",
}
HELD_OUT = ("gouache", "crayon", "autumn", "fog")
ALL_PROMPTS = {**PROMPTS, **MORE_PROMPTS}

# (source, target) captions for the CLIP scores of scripts/eval_student.py, per prompt key
CLIP_CAPTIONS = {
    "oil": ("a photo", "an oil painting with vibrant colors"),
    "anime": ("a photo", "an anime illustration with clean cel shading"),
    "watercolor": ("a photo", "a watercolor painting"),
    "pencil": ("a photo", "a pencil sketch"),
    "ukiyoe": ("a photo", "a ukiyo-e woodblock print"),
    "impressionist": ("a photo", "an impressionist painting"),
    "comic": ("a photo", "a comic book drawing"),
    "charcoal": ("a photo", "a charcoal drawing"),
    "pastel": ("a photo", "a soft pastel drawing"),
    "stainedglass": ("a photo", "a stained glass window"),
    "popart": ("a photo", "a pop art picture"),
    "winter": ("a photo", "a snowy winter scene"),
    "night": ("a photo", "a scene at night"),
    "sunset": ("a photo", "a scene at sunset"),
    "gouache": ("a photo", "a gouache painting"),
    "crayon": ("a photo", "a crayon drawing"),
    "autumn": ("a photo", "an autumn scene with red and orange leaves"),
    "fog": ("a photo", "a foggy morning"),
}

# Training windows come from DAVIS-2017 sequences that dreamlite-stream does not evaluate on
# (assets/davis_heldout20_mp4); evaluation uses four of its 10 tracked clips (assets/davis_mp4).
TRAIN_CLIPS_DIR = "assets/davis_heldout20_mp4"
EVAL_CLIPS_DIR = "assets/davis_mp4"
EVAL_CLIPS = ("parkour", "libby", "blackswan", "dance-twirl")
