"""Tags an image by embedding similarity rather than by generating text.

Every adjective in `vocabulary.ADJECTIVES` is scored against the image and the best
in each facet kept, so the same image always produces the same tags.

SigLIP rather than CLIP because it is trained with a pairwise sigmoid loss, scoring
each image/text pair on its own. CLIP's softmax makes candidate labels compete for a
fixed budget of probability, the wrong shape for ten independent attributes.
"""

import io
import threading

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from vocabulary import ADJECTIVES, QUOTAS, TEMPLATES

MODEL_ID = "google/siglip2-base-patch16-224"

# Filled in on first use: ~375MB of weights, too slow to load at import time.
_model = None
_processor = None
_device = None
_text_vectors: dict[str, torch.Tensor] = {}
_logit_scale = 0.0
_logit_bias = 0.0
_ready = False
_load_lock = threading.Lock()


def _pick_device() -> str:
    # Apple Silicon GPU when available, which is most of a second faster per image.
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embed_vocabulary() -> dict[str, torch.Tensor]:
    """Embeds every adjective once and returns the vectors split back per facet.

    Each word goes through every template and the results are averaged: a single
    phrasing carries the quirks of its own wording, and the mean cancels much of
    that out. One batch per template rather than per facet per template is the same
    arithmetic in a quarter of the forward passes.
    """
    facets = list(ADJECTIVES)
    words = [word for facet in facets for word in ADJECTIVES[facet]]

    per_template = []
    for template in TEMPLATES:
        prompts = [template.format(word) for word in words]
        inputs = _processor(
            text=prompts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(_device)
        with torch.no_grad():
            # This transformers version returns an output object, not a bare tensor.
            vectors = _model.get_text_features(**inputs).pooler_output
        per_template.append(torch.nn.functional.normalize(vectors, dim=-1))

    averaged = torch.nn.functional.normalize(
        torch.stack(per_template).mean(dim=0), dim=-1
    )

    # Slice the flat result back into the facets it came from.
    vectors_by_facet = {}
    start = 0
    for facet in facets:
        end = start + len(ADJECTIVES[facet])
        vectors_by_facet[facet] = averaged[start:end]
        start = end
    return vectors_by_facet


def _embed_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        vector = _model.get_image_features(**inputs).pooler_output
    return torch.nn.functional.normalize(vector, dim=-1)


def _load() -> None:
    """Loads the model and embeds the vocabulary, once, safely under concurrency.

    FastAPI runs the sync `/generate` handler on a threadpool, so two requests could
    reach this together. `_ready` is set last, so nobody sees a half-built state.
    """
    global _model, _processor, _device, _logit_scale, _logit_bias, _ready
    if _ready:
        return

    with _load_lock:
        if _ready:  # another thread finished while we waited for the lock
            return

        _device = _pick_device()
        _model = AutoModel.from_pretrained(MODEL_ID).to(_device).eval()
        _processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Frozen, so read once as plain floats rather than re-exponentiating a
        # tensor per facet per request. no_grad because they still carry a gradient.
        with torch.no_grad():
            _logit_scale = float(_model.logit_scale.exp())
            _logit_bias = float(_model.logit_bias)

        _text_vectors.update(_embed_vocabulary())

        _ready = True


def siglip_tags_by_facet(image_bytes: bytes) -> dict[str, list[tuple[str, float]]]:
    """Every adjective in every facet, ranked best-first, with its score.

    The full ranking rather than just the quota's worth, so a caller (the judge
    replacement flow in main.py) can swap a flagged tag for the next-best word in
    the same facet without re-embedding the image.
    """
    _load()

    image_vector = _embed_image(image_bytes)

    ranked: dict[str, list[tuple[str, float]]] = {}
    for facet, words in ADJECTIVES.items():
        similarity = (image_vector @ _text_vectors[facet].T).squeeze(0)
        # SigLIP trained with a sigmoid over a scaled, shifted cosine, so its own
        # scale and bias turn these (a narrow band around 0.1) into readable
        # probabilities. Monotonic, so it changes the numbers but never the ranking.
        probability = torch.sigmoid(similarity * _logit_scale + _logit_bias)
        order = torch.argsort(probability, descending=True)
        ranked[facet] = [
            (words[int(index)], round(float(probability[index]), 4)) for index in order
        ]
    return ranked


def embedding_projection(image_bytes: bytes, points: list[tuple[str, str]]) -> dict:
    """2D PCA layout of the given (word, facet) points, with the image vector
    projected onto that same basis.

    Illustrative, not analytic: SigLIP's real embedding space is far higher-
    dimensional (768-d here), so collapsing it to a plane necessarily distorts
    relative distances -- two points that land close together on this plot
    aren't guaranteed to be the two closest in the actual space. This exists
    for the tag-analysis scatter plot only; nothing about tag selection or
    ranking reads from it.

    The PCA basis is fit on the word vectors alone, not the image. SigLIP (like
    CLIP) has a well-documented "modality gap" -- image and text embeddings
    occupy visibly separate regions of the space -- so an image vector sits far
    outside the tag cluster on almost every request. Including it in the fit
    would spend both components chasing that one outlier and collapse the 20
    tags, which are the part worth actually resolving, into an unreadable
    smear. The frontend clamps the projected image point to the plot's edge
    when it falls outside the tags' bounds rather than rescaling around it.
    """
    _load()

    image_vector = _embed_image(image_bytes)  # 1 x D
    word_vectors = torch.stack(
        [_text_vectors[facet][ADJECTIVES[facet].index(word)] for word, facet in points]
    )  # N x D

    mean = word_vectors.mean(dim=0, keepdim=True)
    centered_words = word_vectors - mean
    _, _, v = torch.pca_lowrank(centered_words, q=2)
    basis = v[:, :2]

    word_coords = (centered_words @ basis).tolist()
    image_x, image_y = ((image_vector - mean) @ basis).squeeze(0).tolist()

    return {
        "model": MODEL_ID,
        "source_dimensions": word_vectors.shape[-1],
        "image": {"x": image_x, "y": image_y},
        "points": [
            {"tag": word, "facet": facet, "x": x, "y": y}
            for (word, facet), (x, y) in zip(points, word_coords)
        ],
    }


def _ordered_by_score(tags: list[str], scores: dict[str, float]) -> tuple[list[str], dict[str, float]]:
    """Sorts tags best-first and rebuilds the score dict to match that order.

    Stable, so ties fall back to whatever order `tags` was already in.
    """
    tags = sorted(tags, key=lambda word: scores[word], reverse=True)
    return tags, {word: scores[word] for word in tags}


def select_tags(ranked: dict[str, list[tuple[str, float]]]) -> tuple[list[str], dict[str, float]]:
    """Takes each facet's top `QUOTAS[facet]` candidates and orders them by score.

    Filled per facet rather than as a global top 10, which tends to collapse onto
    one idea and its synonyms.
    """
    tags: list[str] = []
    scores: dict[str, float] = {}
    for facet, candidates in ranked.items():
        for word, score in candidates[: QUOTAS[facet]]:
            tags.append(word)
            scores[word] = score

    # The quotas decide *which* ten survive; this orders them by match strength.
    return _ordered_by_score(tags, scores)


def word_to_facet(ranked: dict[str, list[tuple[str, float]]]) -> dict[str, str]:
    """Flattens the ranked-by-facet structure into a word -> facet lookup.

    Shared by `replace_flagged_tags` below and main.py's tagging summary --
    both need to know which facet a given word came from.
    """
    return {word: facet for facet, candidates in ranked.items() for word, _ in candidates}


def replace_flagged_tags(
    ranked: dict[str, list[tuple[str, float]]],
    tags: list[str],
    scores: dict[str, float],
    flagged: set[str],
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Swaps each flagged tag for the next-best unused word in its own facet.

    Same-facet rather than a flat nearest-match, so a flagged mood word is
    replaced by another mood word and the quota mix (color:2, light:1, mood:3,
    texture:2, energy:1, style:1) never skews. If a facet's candidates are all
    already in use, the original tag is kept rather than dropped.

    The third return value maps each flagged tag that was actually swapped to
    its replacement -- callers that surface the judging process (main.py's
    history record) use it to show what replaced what. Tags kept despite being
    flagged (facet exhausted) are omitted from this mapping.
    """
    facet_of = word_to_facet(ranked)
    used = set(tags)

    new_tags: list[str] = []
    new_scores = dict(scores)
    replacements: dict[str, str] = {}
    for tag in tags:
        if tag not in flagged:
            new_tags.append(tag)
            continue

        facet = facet_of[tag]
        replacement = next((c for c in ranked[facet] if c[0] not in used), None)
        if replacement is None:
            # Every candidate in this facet is already in the tag list -- the
            # facet's vocabulary is too small relative to its quota for this
            # image. Visible in the terminal so a systematic case doesn't go
            # unnoticed, same as judge.py's own fallback for a missing verdict.
            print(f"siglip_tagger: facet exhausted, keeping flagged tag {tag!r}")
            new_tags.append(tag)
            continue

        word, score = replacement
        used.add(word)
        new_tags.append(word)
        new_scores[word] = score
        replacements[tag] = word

    new_tags, new_scores = _ordered_by_score(new_tags, new_scores)
    return new_tags, new_scores, replacements


def siglip_tags(image_bytes: bytes) -> tuple[list[str], dict[str, float]]:
    """Returns 10 adjectives describing the image, plus each one's score."""
    return select_tags(siglip_tags_by_facet(image_bytes))
