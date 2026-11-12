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

# What FLUX.2-klein-4B is given instead, where it reads the instruction above differently from VIBE (scripts/
# klein_teacher.py): a garish oil painting, a pastel without chalk, ukiyo-e with a border and a seal. The student is
# still trained on ALL_PROMPTS[key].
KLEIN_PROMPTS = {
    "oil": "transfer this to oil painting style with visible brushstrokes, keep the original colors",
    "pastel": "redraw this as a pastel chalk drawing with rich colors and visible chalky strokes",
    "ukiyoe": "transfer this to ukiyo-e woodblock print style, flat colors and bold outlines, the picture fills the "
              "whole image, no border, no text, no seal",
}

# Broad instructions a student resized to another size with transformer blocks removed is self-distilled on, from the
# full student at that size (scripts/resize_pack.py --mode self), so that it keeps what the full student does beyond
# the instructions it was trained on. None of HELD_OUT.
BROAD_PROMPTS = {
    "cartoon3d": "transfer this to 3D cartoon render style, soft studio lighting",
    "pixelart": "transfer this to pixel art style, limited color palette",
    "mosaic": "transfer this to mosaic style, small colored tiles",
    "inkwash": "transfer this to Chinese ink wash painting style, black ink on rice paper",
    "cyberpunk": "make it cyberpunk, neon pink and blue lights",
    "sepia": "transfer this to an old sepia photograph with film grain",
    "bw": "make it a black and white photo, high contrast",
    "pointillism": "transfer this to pointillism style, tiny dots of pure color",
    "cubism": "transfer this to cubist painting style, fragmented geometric shapes",
    "artnouveau": "transfer this to art nouveau style, flowing ornamental lines",
    "lowpoly": "transfer this to low poly 3D style, flat shaded facets",
    "clay": "transfer this to claymation style, plasticine textures",
    "embroidery": "transfer this to embroidery, colored threads stitched on fabric",
    "graffiti": "transfer this to graffiti street art style, spray paint",
    "colorpencil": "transfer this to a colored pencil drawing on paper",
    "swirl": "transfer this to post-impressionist style, swirling thick brushstrokes",
    "lineart": "transfer this to clean line art, black outlines on white",
    "woodcut": "transfer this to a black and white woodcut print, carved lines",
    "rain": "make it rainy, wet reflective surfaces",
    "spring": "make it spring, fresh green leaves and blossoms",
    "overcast": "make it overcast, soft grey daylight",
    "underwater": "make it look underwater, blue-green tint",
    "moonlight": "make it a moonlit night, cold blue light",
    "cinematic": "give it a cinematic teal and orange color grade",
    "vhs": "make it look like an old VHS recording, washed-out colors and scan lines",
}

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
