"""LLM-as-judge: asks Claude Sonnet 5 (vision) whether a SigLIP tag actually fits
the image it was chosen for.

Shared by the live app (main.py's /generate, to swap out a bad tag before it reaches
the prompt) and the offline eval script (eval/judge.py, to report how often SigLIP's
picks hold up). Imported lazily, same convention as embedding_tagger, since anthropic
and its network client aren't needed until a request actually calls the judge.
"""

import base64
import io

from PIL import Image

from judge_client import call_judge

# Vision models don't benefit from full-resolution originals and a smaller payload
# means a faster round trip. 1024px on the long side is comfortably above what the
# model actually uses for this kind of aesthetic judgment.
JUDGE_MAX_SIDE = 1024

# One-word adjectives, grouped by eight facets in vocabulary.py, is given to the judge
# as framing so it understands *why* a tag might be a weak fit (it was chosen by
# embedding similarity against a fixed vocabulary, not written by a human for this
# specific image) without leaking which facet each tag came from -- that's an
# implementation detail the judge doesn't need to do its job.
JUDGE_PROMPT_TEMPLATE = """You are reviewing the output of an automated image-tagging system.

The system looked at this image and picked these {tag_count} one-word adjectives -- \
covering color, lighting, mood, emotional valence, emotional arousal, texture, energy, \
and visual style -- to feed a music generator that will turn this image into a song:

{tag_list}

Each word was chosen because it scored highest, by embedding similarity, against a \
fixed vocabulary list -- not because a human curated it for this specific image. As a \
result, some words can be a poor or even nonsensical fit.

For each of the 10 words, decide whether it captures the mood or feeling the image \
evokes -- not whether it's a literal, physically accurate description. These are \
evocative, mood-board-style words for a musician, not literal captions, so a word does \
not need to match the image's literal physical state to be a good fit. For example, \
"swaying" for a photo of completely still leaves is a good fit if the leaves' shape, \
arrangement, or overall feel evokes a swaying rhythm -- the tag is describing the mood \
the image conjures, not asserting the leaves are literally in motion. Give the tag the \
benefit of the doubt -- more than you might instinctively want to. Only flag a word when \
it is flatly contradicted by the image's mood, to the point that almost any viewer would \
call it simply wrong -- e.g. "aquatic" for a desert photo, or "joyful" for a scene that \
is unambiguously grim. Being static, showing the "wrong" literal color, or otherwise not \
literally depicting the word is NOT, on its own, a reason to flag. If you're genuinely \
unsure whether a word fits, do not flag it.

Return a verdict for all {tag_count} words, in the same order they were given. For each word, \
set "flagged" to true only if it fails the mood standard above, and false otherwise. \
When "flagged" is true, give a one-sentence "reason" explaining, in terms of the \
image's actual mood or feeling, why the word doesn't fit. When "flagged" is false, set \
"reason" to an empty string."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "flagged": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["tag", "flagged", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def _resize(image_bytes: bytes, max_side: int, format: str) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, format=format, quality=90 if format == "JPEG" else None)
    return out.getvalue()


def judge_image(image_bytes: bytes, tags: list[str]) -> list[dict]:
    """Sends one image + its 10 tags to Claude Sonnet 5 in a single call.

    Returns a list of {tag, flagged, reason} in the same order as `tags`. Falls
    back to an unflagged verdict for any tag the model's response is missing or
    mangled for, rather than silently dropping it or crashing the whole run.
    """
    judge_bytes = _resize(image_bytes, JUDGE_MAX_SIDE, "JPEG")
    image_b64 = base64.b64encode(judge_bytes).decode("ascii")

    tag_list = "\n".join(f"- {tag}" for tag in tags)
    prompt = JUDGE_PROMPT_TEMPLATE.format(tag_list=tag_list, tag_count=len(tags))

    parsed = call_judge(
        max_tokens=2048,
        schema=JUDGE_SCHEMA,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    verdicts_by_tag = {v["tag"]: v for v in parsed.get("verdicts", []) if "tag" in v}

    result = []
    for tag in tags:
        verdict = verdicts_by_tag.get(tag)
        if verdict is None:
            # Model dropped or renamed a tag in its response -- treat as unflagged
            # rather than failing the whole request.
            print(f"judge: response missing verdict for tag {tag!r}")
            result.append({"tag": tag, "flagged": False, "reason": ""})
        else:
            result.append(
                {
                    "tag": tag,
                    "flagged": bool(verdict.get("flagged", False)),
                    "reason": verdict.get("reason", "") or "",
                }
            )
    return result
