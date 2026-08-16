"""The adjective vocabulary the SigLIP tagger chooses from.

An embedding tagger can only return words from a fixed list, so this file is a hard
ceiling on what can ever be said about an image. Adding a word here is the only way
to make it sayable.

These tags become the prompt for a music generator, so prefer words describing
texture, motion, weight and energy. A song can be brittle or swaying; it cannot be
purple or striped.

Grouped by facet so tags can be picked with a per-facet quota. Without that, scoring
every word independently tends to return one concept and all its synonyms (warm,
warmth, hot, golden, sunny...).
"""

ADJECTIVES = {
    "colour": [
        "crimson", "scarlet", "amber", "golden", "ochre", "russet", "sepia",
        "teal", "cyan", "azure", "cobalt", "indigo", "violet", "magenta",
        "rosy", "coral", "peach", "olive", "emerald", "jade", "mint",
        "mustard", "lavender", "charcoal", "slate", "ashen", "ivory",
        "cream", "beige", "taupe", "bronze", "copper", "silvery", "pearlescent",
        "monochrome", "pastel", "neon", "fluorescent", "saturated",
        "desaturated", "muted", "vivid", "washed", "bleached", "inky",
        "verdant", "earthy", "icy",
    ],
    "light": [
        "bright", "dim", "glowing", "luminous", "radiant", "backlit", "shadowy",
        "gloomy", "murky", "dappled", "glaring", "blinding", "soft", "harsh",
        "diffuse", "hazy", "misty", "overcast", "sunlit", "moonlit",
        "candlelit", "flickering", "shimmering", "sparkling",
        "dusky", "twilit", "silhouetted", "contrasty", "flat", "dark",
        "blazing", "faint", "smoky", "starlit", "burnished", "cloudy",
        "gilded", "shaded",
    ],
    # Qualitative/character words only -- a specific flavor of feeling that a
    # position on the valence and arousal axes below can't express on its own
    # (e.g. "bittersweet" is literally mixed-valence; "eerie" is about a kind
    # of ambiguous threat, not just low valence + moderate arousal). Words that
    # were mostly doing "how positive" or "how activated" duty moved to the
    # `valence`/`arousal` facets instead of staying here as near-duplicates.
    "mood": [
        "wistful", "playful", "whimsical", "ethereal", "mystical", "eerie",
        "haunting", "ominous", "menacing", "lonely",
        "intimate", "tender", "romantic", "nostalgic", "bittersweet",
        "defiant", "brooding", "solemn", "reverent", "yearning", "sullen",
        "wary", "sober", "wondrous", "desolate",
    ],
    # Positive <-> negative emotional tone. Deliberately spans both ends of
    # the axis in one list, since the image -- not this file -- decides which
    # end wins for a given photo.
    "valence": [
        "joyful", "hopeful", "carefree", "buoyant", "warm",
        "content", "elated",
        "melancholy", "sombre", "bitter", "grim",
        "wretched", "dismal",
    ],
    # Calm <-> excited emotional activation, independent of valence (an image
    # can be excited-and-joyful or excited-and-anxious). Kept distinct from
    # `energy`'s vocabulary, which is about the image's visual/kinetic motion
    # rather than a felt emotional intensity -- SigLIP's own embedding space
    # doesn't respect that distinction as cleanly (see the Phase 1 facet-check
    # report), so several arousal words that score close to energy words
    # (electric, charged, tense...) stayed here deliberately: they're felt-
    # intensity words even where SigLIP finds them close to motion words.
    "arousal": [
        "serene", "calm", "sedate",
        "hushed", "meditative", "dreamy", "blissful",
        "electric", "charged", "exhilarated", "euphoric", "tense", "anxious",
        "wired", "overwrought",
    ],
    "texture": [
        "grainy", "gritty", "smooth", "glossy", "matte", "velvety", "silky",
        "coarse", "rough", "weathered", "worn", "cracked", "peeling", "rusted",
        "polished", "crisp", "blurry", "smudged", "streaked", "speckled",
        "mottled", "marbled", "woven", "fibrous", "powdery", "chalky", "waxy",
        "glassy", "metallic", "wooden", "stony", "fluffy", "downy", "prickly",
        "jagged", "pitted", "frosted", "dusty", "damp",
        "brittle", "supple", "grimy",
    ],
    "energy": [
        "languid", "sluggish", "still", "motionless", "drifting",
        "flowing", "swirling", "churning", "frantic", "frenetic", "hectic",
        "bustling", "driving", "pulsing", "throbbing", "surging", "explosive",
        "agitated", "jittery", "brisk", "lively", "energetic",
        "dynamic", "kinetic", "sweeping", "cascading", "plunging", "soaring",
        "gentle", "steady", "relentless", "halting", "suspended", "poised",
        "coiled", "unhurried", "urgent", "swaying",
    ],
    "style": [
        "retro", "vintage", "antique", "futuristic", "modern", "minimal",
        "maximal", "baroque", "ornate", "austere", "industrial", "rustic",
        "urban", "pastoral", "cinematic", "painterly", "sketchy", "graphic",
        "geometric", "organic", "abstract", "surreal", "photographic",
        "documentary", "editorial", "kitsch", "psychedelic", "gothic",
        "brutalist", "bohemian", "folkloric", "cyberpunk", "impressionistic",
        "expressionistic", "naturalistic", "stylised", "cartoonish", "glitchy",
        "analog", "digital", "utilitarian", "theatrical",
    ],
}

# How many tags to take from each facet. Sums to 12, weighted toward
# mood/valence/arousal together (5 of 12) because that's what most shapes how
# a song feels. valence and arousal each get a guaranteed slot so a tag list
# can't drift to covering only one axis of feeling (e.g. three mood words
# that are all low-arousal, saying nothing about how activated the image is).
QUOTAS = {
    "colour": 2,
    "light": 1,
    "mood": 3,
    "valence": 1,
    "arousal": 1,
    "texture": 2,
    "energy": 1,
    "style": 1,
}

# Encoding a bare word underperforms a short phrase on CLIP-family models, which
# were trained on captions rather than isolated labels. Each adjective is embedded
# through all of these and the results averaged.
TEMPLATES = [
    "a photo that feels {}",
    "a {} image",
    "{} atmosphere",
    "a picture with a {} quality",
]
