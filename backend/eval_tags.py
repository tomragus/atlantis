"""Phases 2 and 3 of vocabulary_optimizer.md: redundancy margin + coverage score.

For every image in test-images/, runs the tagger and records, per facet, the
margin between the top-1 and top-2 word's probability (Phase 2: a facet whose
margins cluster near zero is crowded with near-synonyms) and the top-1 word's
absolute probability (Phase 3: a facet whose winner still scores low is
missing a concept the corpus actually needs). No judge step -- SigLIP margin
and score data only. Command-line dev tool, not called by the app; run
manually from backend/.

Usage: backend/.venv/bin/python eval_tags.py
"""

import glob
import os
import statistics

from siglip_tagger import siglip_tags_by_facet

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-images")


def main() -> None:
    paths = sorted(
        p
        for ext in ("*.png", "*.jpg", "*.jpeg")
        for p in glob.glob(os.path.join(CORPUS_DIR, ext))
    )
    print(f"Running {len(paths)} images from {CORPUS_DIR}\n")

    margins: dict[str, list[float]] = {}
    top1_scores: dict[str, list[float]] = {}

    for path in paths:
        with open(path, "rb") as f:
            image_bytes = f.read()
        ranked = siglip_tags_by_facet(image_bytes)
        for facet, candidates in ranked.items():
            top1 = candidates[0][1]
            top2 = candidates[1][1]
            margins.setdefault(facet, []).append(top1 - top2)
            top1_scores.setdefault(facet, []).append(top1)

    print(f"{'facet':<10} {'mean margin':>12} {'median margin':>14} {'mean top1':>10} {'min top1':>9}")
    for facet in margins:
        m = margins[facet]
        s = top1_scores[facet]
        print(
            f"{facet:<10} {statistics.mean(m):>12.4f} {statistics.median(m):>14.4f} "
            f"{statistics.mean(s):>10.4f} {min(s):>9.4f}"
        )

    print("\nFacets sorted by mean margin (lowest first -- most redundant):")
    for facet, vals in sorted(margins.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"  {facet:<10} mean margin {statistics.mean(vals):.4f}")

    print("\nFacets sorted by mean top1 score (lowest first -- weakest coverage):")
    for facet, vals in sorted(top1_scores.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"  {facet:<10} mean top1 {statistics.mean(vals):.4f}, min {min(vals):.4f}")


if __name__ == "__main__":
    main()
