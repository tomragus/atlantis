"""LLM-as-judge, pass two: asks Claude Sonnet 5 whether any two of the final tags
describe contradicting moods, since each tag is scored per-facet and independently
of the others -- nothing stops "joyful" (valence) and "haunting" (mood) from both
landing in the same 10 words.

Runs after judge.py's per-tag mood check, on the tags it already settled. Text-only:
whether two words contradict each other is a property of the words themselves, not
the image, so this skips the vision call and its resize/base64 path entirely.
Imported lazily by main.py, same convention as judge.py.
"""

from judge_client import call_judge

CONTRADICTION_PROMPT_TEMPLATE = """You are reviewing a set of one-word adjective tags chosen by an automated image-tagging system, grouped across eight facets (color, light, mood, emotional valence, emotional arousal, texture, energy, visual style), that will be fed into a music generator as mood-board-style descriptors for a song.

Here are the {tag_count} tags, each labeled with the facet it came from:

{tag_list}

Each facet was scored independently, so it's possible for two tags from different \
facets to describe contradictory moods -- for example "joyful" (valence) and \
"haunting" (mood) can't both genuinely describe the same image's feeling. Look \
across all {tag_count} words for any such pairs: two tags a listener would find \
mood-contradictory if both showed up in the same song description, not just tags \
that are merely different or unrelated. Most tag sets will have zero such pairs, \
occasionally one -- do not invent a pair that isn't a genuine mood contradiction.

For each contradicting pair you find, decide which of the two words should be \
dropped: pick the one that is more redundant given the rest of the list \
-- specifically, if the OTHER word in the pair already has a close synonym or \
near-duplicate sitting elsewhere in the list, that other word is the one covered \
twice over, so drop the word that duplicates it least, i.e. the word whose own \
counterpart in the pair is better covered elsewhere. If the other word in the pair \
is the only tag covering its particular shade of feeling, keep it and drop its \
counterpart instead. If neither is clearly more redundant, drop the more generic \
of the two.

Return every contradicting pair you find. For each, give "tag_a" and "tag_b" (the \
two contradicting words, in the order given above), "drop" (exactly one of tag_a \
or tag_b -- the one to replace), and a one-sentence "reason" explaining the \
contradiction and why that one was chosen to drop."""

CONTRADICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag_a": {"type": "string"},
                    "tag_b": {"type": "string"},
                    "drop": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["tag_a", "tag_b", "drop", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["contradictions"],
    "additionalProperties": False,
}


def judge_contradictions(tags: list[str], facets: dict[str, str]) -> list[dict]:
    """Sends the tag list (no image) to Claude Sonnet 5 and asks it to find pairs
    that describe contradicting moods, and which of each pair is more redundant
    given the rest of the list.

    Returns a list of {tag_a, tag_b, drop, reason}, one entry per genuine
    contradiction found (often empty). Any entry referencing a tag that isn't
    actually in `tags`, or a "drop" that's neither tag_a nor tag_b, is dropped
    rather than allowed to crash the swap step in main.py.
    """
    tag_list = "\n".join(f"- {tag} ({facets.get(tag, 'unknown')})" for tag in tags)
    prompt = CONTRADICTION_PROMPT_TEMPLATE.format(tag_list=tag_list, tag_count=len(tags))

    parsed = call_judge(
        max_tokens=1024,
        schema=CONTRADICTION_SCHEMA,
        messages=[{"role": "user", "content": prompt}],
    )

    tag_set = set(tags)
    contradictions = []
    for item in parsed.get("contradictions", []):
        tag_a, tag_b, drop = item.get("tag_a"), item.get("tag_b"), item.get("drop")
        if tag_a not in tag_set or tag_b not in tag_set or drop not in (tag_a, tag_b):
            print(f"contradiction_judge: skipping malformed entry {item!r}")
            continue
        contradictions.append(
            {
                "tag_a": tag_a,
                "tag_b": tag_b,
                "drop": drop,
                "reason": item.get("reason", "") or "",
            }
        )
    return contradictions
