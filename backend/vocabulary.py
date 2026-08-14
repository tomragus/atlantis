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
        "mustard", "lilac", "lavender", "charcoal", "slate", "ashen", "ivory",
        "cream", "beige", "taupe", "bronze", "copper", "silvery", "pearlescent",
        "monochrome", "greyscale", "pastel", "neon", "fluorescent", "saturated",
        "desaturated", "muted", "vivid", "washed", "bleached", "inky",
        "verdant", "earthy", "icy",
    ],
    "light": [
        "bright", "dim", "glowing", "luminous", "radiant", "backlit", "shadowy",
        "gloomy", "murky", "dappled", "glaring", "blinding", "soft", "harsh",
        "diffuse", "hazy", "misty", "foggy", "overcast", "sunlit", "moonlit",
        "candlelit", "flickering", "shimmering", "gleaming", "sparkling",
        "dusky", "twilit", "silhouetted", "contrasty", "flat", "dark",
        "blazing", "faint", "smoky", "starlit", "burnished", "cloudy",
        "gilded", "shaded",
    ],
    "mood": [
        "serene", "tranquil", "peaceful", "calm", "meditative", "melancholy",
        "wistful", "forlorn", "sombre", "mournful", "joyful", "jubilant",
        "cheerful", "playful", "whimsical", "dreamy", "ethereal", "mystical",
        "eerie", "haunting", "ominous", "menacing", "tense", "anxious",
        "unsettling", "lonely", "desolate", "intimate", "tender", "romantic",
        "nostalgic", "bittersweet", "hopeful", "triumphant", "defiant",
        "brooding", "contemplative", "solemn", "reverent", "uneasy",
        "yearning", "blissful", "euphoric", "sullen", "wary", "sober",
        "carefree", "wondrous", "grim",
    ],
    "texture": [
        "grainy", "gritty", "smooth", "glossy", "matte", "velvety", "silky",
        "coarse", "rough", "weathered", "worn", "cracked", "peeling", "rusted",
        "polished", "crisp", "blurry", "smudged", "streaked", "speckled",
        "mottled", "marbled", "woven", "fibrous", "powdery", "chalky", "waxy",
        "glassy", "metallic", "wooden", "stony", "fluffy", "downy", "prickly",
        "jagged", "sleek", "pitted", "frosted", "dusty", "damp", "slick",
        "brittle", "supple", "grimy",
    ],
    "energy": [
        "languid", "sluggish", "still", "static", "motionless", "drifting",
        "flowing", "swirling", "churning", "frantic", "frenetic", "hectic",
        "bustling", "driving", "pulsing", "throbbing", "surging", "explosive",
        "restless", "agitated", "jittery", "brisk", "lively", "energetic",
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
        "analog", "digital", "ornamental", "utilitarian", "theatrical",
    ],
}

# How many tags to take from each facet. Sums to 10, weighted toward mood because
# that is what most shapes how a song feels.
QUOTAS = {
    "colour": 2,
    "light": 1,
    "mood": 3,
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
