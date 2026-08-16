"""Phase 1 of vocabulary_optimizer.md: facet-consistency check.

Embeds every word in vocabulary.ADJECTIVES (the same way siglip_tagger already
does) and, for each word, finds its nearest neighbors across the *entire*
vocabulary rather than just its own facet. A word whose neighbors mostly live
in a different facet is a candidate for reassignment -- reported as a
before/after pair, never applied automatically. Command-line dev tool, not
called by the app; run manually from backend/.

Usage: backend/.venv/bin/python check_facets.py
"""

import torch

import siglip_tagger as st
from vocabulary import ADJECTIVES

TOP_K = 5


def main() -> None:
    st._load()  # populates st._text_vectors without needing an image

    words: list[str] = []
    facets: list[str] = []
    vectors: list[torch.Tensor] = []
    for facet, vecs in st._text_vectors.items():
        for i, word in enumerate(ADJECTIVES[facet]):
            words.append(word)
            facets.append(facet)
            vectors.append(vecs[i])

    matrix = torch.stack(vectors)  # (N, dim), already unit-normalized
    similarity = matrix @ matrix.T  # (N, N) cosine similarity

    flagged = []
    for i, word in enumerate(words):
        own_facet = facets[i]
        order = torch.argsort(similarity[i], descending=True)
        neighbor_indices = [int(j) for j in order if int(j) != i][:TOP_K]
        neighbor_facets = [facets[j] for j in neighbor_indices]
        neighbor_words = [words[j] for j in neighbor_indices]

        majority_facet = max(set(neighbor_facets), key=neighbor_facets.count)
        majority_count = neighbor_facets.count(majority_facet)

        if majority_facet != own_facet and majority_count > TOP_K // 2:
            flagged.append(
                {
                    "word": word,
                    "current_facet": own_facet,
                    "suggested_facet": majority_facet,
                    "neighbors": list(zip(neighbor_words, neighbor_facets)),
                }
            )

    print(f"{len(words)} words embedded across {len(ADJECTIVES)} facets.")
    print(f"{len(flagged)} words flagged for possible facet reassignment:\n")
    for f in flagged:
        print(f"  {f['word']!r}: {f['current_facet']} -> {f['suggested_facet']}")
        for nw, nf in f["neighbors"]:
            marker = "*" if nf == f["suggested_facet"] else " "
            print(f"      {marker} {nw} ({nf})")
    if not flagged:
        print("  (none)")


if __name__ == "__main__":
    main()
