"""Tags an image by embedding similarity rather than by generating text.

Every adjective in `vocabulary.ADJECTIVES` is scored against the image and the best
in each facet kept, so the same image always produces the same tags. This module
supports five embedding algorithms (SigLIP, CLIP, ALIGN, MetaCLIP, CoCa) through one
shared tagging/PCA code path, keyed by an `algorithm` argument -- each keeps its own
lazily-loaded model/processor/vocabulary-vectors, mirroring the single-model pattern
this file started as.

SigLIP is the default rather than CLIP because it is trained with a pairwise sigmoid
loss, scoring each image/text pair on its own. CLIP's softmax makes candidate labels
compete for a fixed budget of probability, the wrong shape for ten independent
attributes. The other three HF algorithms (CLIP, ALIGN, MetaCLIP) and CoCa are offered
alongside it to show off how each model's embedding space differs, not to out-score
it -- so, per plan, none of them get a calibrated probability transform: only SigLIP
has a trained logit_scale/logit_bias to turn cosine similarity into one, so the other
four score with raw cosine similarity on their own native scale.
"""

import io
import threading
from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from algorithms import ALGORITHMS, DEFAULT_ALGORITHM
from vocabulary import ADJECTIVES, QUOTAS, TEMPLATES

# HF checkpoints loaded through the same AutoModel/AutoProcessor interface SigLIP
# already used -- get_text_features/get_image_features both return a
# BaseModelOutputWithPooling under transformers==5.14.1 for all three of these, same
# as SigLIP, so `_embed_image`/`_embed_vocabulary` below read `.pooler_output` for
# every HF algorithm without a per-model branch.
HF_MODEL_IDS = {
    "siglip": "google/siglip2-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
    "align": "kakaobrain/align-base",
    "metaclip": "facebook/metaclip-b32-400m",
}

# CoCa isn't in transformers -- it needs open_clip's own create/tokenize/encode API,
# loaded only when this algorithm is actually requested (see _load below).
COCA_MODEL_NAME = "coca_ViT-B-32"
# The mscoco-finetuned checkpoint rather than the base contrastive one: it's the
# variant open_clip trains specifically for the captioning head this app uses.
COCA_PRETRAINED = "mscoco_finetuned_laion2b_s13b_b90k"

# Human-readable model id per algorithm, surfaced in embedding_projection's "model"
# field and (via main.py's GET /algorithms) to the frontend.
MODEL_IDS = {**HF_MODEL_IDS, "coca": f"open_clip:{COCA_MODEL_NAME}/{COCA_PRETRAINED}"}


@dataclass
class _AlgorithmState:
    """Everything one algorithm needs, loaded lazily behind its own lock.

    Was a single set of module-level globals when this file only supported SigLIP;
    replaced with one instance per algorithm so five models' weights/processors never
    collide, each still loading -- and locking -- independently of the others.
    """

    model: object = None
    device: str | None = None
    text_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    ready: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    # HF algorithms only.
    processor: object = None
    logit_scale: float = 0.0
    logit_bias: float = 0.0
    # CoCa only.
    tokenizer: object = None
    preprocess: object = None


_state: dict[str, _AlgorithmState] = {name: _AlgorithmState() for name in ALGORITHMS}


def _pick_device() -> str:
    # Apple Silicon GPU when available, which is most of a second faster per image.
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embed_vocabulary(algorithm: str) -> dict[str, torch.Tensor]:
    """Embeds every adjective once and returns the vectors split back per facet.

    Each word goes through every template and the results are averaged: a single
    phrasing carries the quirks of its own wording, and the mean cancels much of
    that out. One batch per template rather than per facet per template is the same
    arithmetic in a quarter of the forward passes.
    """
    state = _state[algorithm]
    facets = list(ADJECTIVES)
    words = [word for facet in facets for word in ADJECTIVES[facet]]

    per_template = []
    for template in TEMPLATES:
        prompts = [template.format(word) for word in words]
        if algorithm == "coca":
            tokens = state.tokenizer(prompts).to(state.device)
            with torch.no_grad():
                vectors = state.model.encode_text(tokens, normalize=False)
        else:
            inputs = state.processor(
                text=prompts, padding="max_length", truncation=True, return_tensors="pt"
            ).to(state.device)
            with torch.no_grad():
                # This transformers version returns an output object, not a bare tensor.
                vectors = state.model.get_text_features(**inputs).pooler_output
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


def _open_image(image_bytes: bytes) -> Image.Image:
    """Decodes uploaded bytes into an RGB PIL image, converting away any alpha/
    palette channel every embedding model's preprocessing expects not to see.

    Shared by `_embed_image` and `generate_caption` -- both start from the same
    raw upload.
    """
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _embed_image(algorithm: str, image_bytes: bytes) -> torch.Tensor:
    state = _state[algorithm]
    image = _open_image(image_bytes)
    if algorithm == "coca":
        pixels = state.preprocess(image).unsqueeze(0).to(state.device)
        with torch.no_grad():
            vector = state.model.encode_image(pixels, normalize=False)
    else:
        inputs = state.processor(images=image, return_tensors="pt").to(state.device)
        with torch.no_grad():
            vector = state.model.get_image_features(**inputs).pooler_output
    return torch.nn.functional.normalize(vector, dim=-1)


def _patch_open_clip_generate(open_clip) -> None:
    """Works around an open_clip/transformers version mismatch in CoCa's `.generate()`.

    open_clip.coca_model imports `BeamSearchScorer` from `transformers` at module
    load time and, if that single name fails to import, swallows the ImportError and
    disables *all* of `.generate()` -- including the top_k/top_p decoding paths,
    which never touch `BeamSearchScorer` at runtime. `transformers==5.14.1` (pinned
    for the other four algorithms' `.pooler_output` shape -- see requirements.txt)
    has removed `BeamSearchScorer` entirely, so CoCa's captioning would otherwise be
    dead on arrival under this repo's pin. Re-supplies the working top_k/top_p
    classes open_clip already successfully imported before hitting the missing name,
    so `generate(..., generation_type="top_k")` (used by `generate_caption` below,
    which never requests beam search) works without touching the pin.
    """
    from transformers import TopKLogitsWarper, TopPLogitsWarper

    coca_model = open_clip.coca_model
    if coca_model._has_transformers:
        return
    coca_model.GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search",
    }
    coca_model._has_transformers = True


def _load(algorithm: str) -> None:
    """Loads one algorithm's model and embeds the vocabulary, once, safely under
    concurrency.

    FastAPI runs sync handlers on a threadpool, so two requests could reach this
    together for the same (or different) algorithm. Each algorithm gets its own
    state and lock, so loading CLIP never blocks a concurrent SigLIP request.
    `ready` is set last, so nobody sees a half-built state.
    """
    state = _state[algorithm]
    if state.ready:
        return

    with state.lock:
        if state.ready:  # another thread finished while we waited for the lock
            return

        state.device = _pick_device()

        if algorithm == "coca":
            # Deferred: open_clip is a second, heavier ML dependency only needed
            # when CoCa is actually picked.
            import open_clip

            _patch_open_clip_generate(open_clip)

            model, _, preprocess = open_clip.create_model_and_transforms(
                COCA_MODEL_NAME, pretrained=COCA_PRETRAINED
            )
            state.model = model.to(state.device).eval()
            state.preprocess = preprocess
            state.tokenizer = open_clip.get_tokenizer(COCA_MODEL_NAME)
        else:
            model_id = HF_MODEL_IDS[algorithm]
            state.model = AutoModel.from_pretrained(model_id).to(state.device).eval()
            state.processor = AutoProcessor.from_pretrained(model_id)

            if algorithm == "siglip":
                # Frozen, so read once as plain floats rather than re-exponentiating
                # a tensor per facet per request. no_grad because they still carry a
                # gradient. Only SigLIP has this calibration head -- CLIP/ALIGN/
                # MetaCLIP score with raw cosine similarity instead (see module
                # docstring).
                with torch.no_grad():
                    state.logit_scale = float(state.model.logit_scale.exp())
                    state.logit_bias = float(state.model.logit_bias)

        state.text_vectors.update(_embed_vocabulary(algorithm))
        state.ready = True


def embed_image(image_bytes: bytes, algorithm: str = DEFAULT_ALGORITHM) -> torch.Tensor:
    """Loads the algorithm (if needed) and embeds the image, once.

    A request that needs both `tags_by_facet` and `embedding_projection` for
    the same image (main.py's /tag route does, for every algorithm) can
    compute the vector here once and pass it into both via their
    `image_vector` parameter, instead of paying for a second forward pass
    through the model for the same image.
    """
    _load(algorithm)
    return _embed_image(algorithm, image_bytes)


def tags_by_facet(
    image_bytes: bytes,
    algorithm: str = DEFAULT_ALGORITHM,
    image_vector: torch.Tensor | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Every adjective in every facet, ranked best-first, with its score.

    The full ranking rather than just the quota's worth, so main.py's tagging
    summary can show the next-best runners-up (the "top 20 candidates" list
    and the embedding plot) alongside whichever tags actually survived the
    judges, without re-embedding the image.

    `image_vector` lets a caller that already embedded this image (via
    `embed_image`) skip doing it again; if omitted, this embeds it itself,
    same as before.
    """
    _load(algorithm)
    state = _state[algorithm]

    if image_vector is None:
        image_vector = _embed_image(algorithm, image_bytes)

    ranked: dict[str, list[tuple[str, float]]] = {}
    for facet, words in ADJECTIVES.items():
        similarity = (image_vector @ state.text_vectors[facet].T).squeeze(0)
        if algorithm == "siglip":
            # SigLIP trained with a sigmoid over a scaled, shifted cosine, so its own
            # scale and bias turn these (a narrow band around 0.1) into readable
            # probabilities. Monotonic, so it changes the numbers but never the
            # ranking.
            score = torch.sigmoid(similarity * state.logit_scale + state.logit_bias)
        else:
            # No trained calibration head for these algorithms (per plan decision):
            # raw cosine similarity on the model's own scale, un-normalized further.
            score = similarity
        order = torch.argsort(score, descending=True)
        # Full precision, not rounded to a fixed number of decimals: rounding to 4
        # places here would silently floor any score below 0.00005 to exactly 0.0,
        # which is indistinguishable from "no match at all" by the time it reaches
        # the frontend. The frontend's own formatScore() decides how to display a
        # tiny-but-nonzero value (scientific notation below 0.001).
        ranked[facet] = [
            (words[int(index)], float(score[index])) for index in order
        ]
    return ranked


def embedding_projection(
    image_bytes: bytes,
    points: list[tuple[str, str]],
    algorithm: str = DEFAULT_ALGORITHM,
    image_vector: torch.Tensor | None = None,
) -> dict:
    """2D PCA layout of the given (word, facet) points, with the image vector
    projected onto that same basis.

    Illustrative, not analytic: every one of these models' real embedding spaces is
    far higher-dimensional than 2, so collapsing it to a plane necessarily distorts
    relative distances -- two points that land close together on this plot aren't
    guaranteed to be the two closest in the actual space. This exists for the
    tag-analysis scatter plot only; nothing about tag selection or ranking reads
    from it.

    The PCA basis is fit on the word vectors alone, not the image. SigLIP-family and
    CLIP-family models share a well-documented "modality gap" -- image and text
    embeddings occupy visibly separate regions of the space -- so an image vector
    sits far outside the tag cluster on almost every request, for every algorithm
    here. Including it in the fit would spend both components chasing that one
    outlier and collapse the 20 tags, which are the part worth actually resolving,
    into an unreadable smear. The frontend clamps the projected image point to the
    plot's edge when it falls outside the tags' bounds rather than rescaling
    around it.

    `image_vector` lets a caller that already embedded this image (via
    `embed_image`) skip doing it again; if omitted, this embeds it itself,
    same as before.
    """
    _load(algorithm)
    state = _state[algorithm]

    if image_vector is None:
        image_vector = _embed_image(algorithm, image_bytes)  # 1 x D
    word_vectors = torch.stack(
        [
            state.text_vectors[facet][ADJECTIVES[facet].index(word)]
            for word, facet in points
        ]
    )  # N x D

    mean = word_vectors.mean(dim=0, keepdim=True)
    centered_words = word_vectors - mean
    # Exact SVD, not `torch.pca_lowrank`: that function is a randomized
    # approximation (Halko et al.) that draws its own unseeded `torch.randn`
    # projection matrix internally, so the same 20 tag vectors could plot at a
    # different rotation on every request. With only ~20 points, exact SVD is
    # cheap and gives the same top-2 directions deterministically every time.
    _, _, vh = torch.linalg.svd(centered_words, full_matrices=False)
    basis = vh[:2].T

    word_coords = (centered_words @ basis).tolist()
    image_x, image_y = ((image_vector - mean) @ basis).squeeze(0).tolist()

    return {
        "model": MODEL_IDS[algorithm],
        "source_dimensions": word_vectors.shape[-1],
        "image": {"x": image_x, "y": image_y},
        "points": [
            {"tag": word, "facet": facet, "x": x, "y": y}
            for (word, facet), (x, y) in zip(points, word_coords)
        ],
    }


def generate_caption(image_bytes: bytes) -> str:
    """CoCa-only: a free-text caption from its captioning head.

    Display-only, per plan -- never touches tag selection or the Suno prompt. Uses
    the same lazily-loaded CoCa state as `tags_by_facet`/`embedding_projection`, so
    calling this after (or before) either of those doesn't reload anything.
    """
    _load("coca")
    state = _state["coca"]

    import open_clip

    image = _open_image(image_bytes)
    pixels = state.preprocess(image).unsqueeze(0).to(state.device)
    with torch.no_grad():
        # top_k rather than the default beam_search: see _patch_open_clip_generate
        # for why beam_search's BeamSearchScorer isn't available under this repo's
        # pinned transformers version.
        generated = state.model.generate(pixels, generation_type="top_k")
    caption = open_clip.decode(generated[0]).split("<end_of_text>")[0]
    return caption.replace("<start_of_text>", "").strip()


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

    Used by main.py's tagging summary to label each tag with the facet it
    came from.
    """
    return {word: facet for facet, candidates in ranked.items() for word, _ in candidates}


def drop_flagged_tags(
    tags: list[str],
    scores: dict[str, float],
    flagged: set[str],
) -> tuple[list[str], dict[str, float]]:
    """Removes every flagged tag from the list -- no replacement is picked.

    A judge flagging a word is a statement that the word doesn't belong, not
    that some other word from the same facet does; dropping it outright keeps
    the tag list honest about what's actually a good fit, at the cost of a
    shorter (and per-facet, uneven) final list. No floor on how few tags can
    remain -- if every tag gets flagged across both judges, the prompt sent to
    Suno just has fewer (or no) tags.
    """
    new_tags = [tag for tag in tags if tag not in flagged]
    new_scores = {tag: scores[tag] for tag in new_tags}
    return new_tags, new_scores


def siglip_tags_by_facet(image_bytes: bytes) -> dict[str, list[tuple[str, float]]]:
    """SigLIP-only wrapper over `tags_by_facet`, kept for `eval/judge.py`'s import."""
    return tags_by_facet(image_bytes, algorithm="siglip")


def siglip_tags(image_bytes: bytes) -> tuple[list[str], dict[str, float]]:
    """Returns 10 adjectives describing the image, plus each one's score."""
    return select_tags(siglip_tags_by_facet(image_bytes))
