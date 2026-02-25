#!/usr/bin/env python3
"""
Corpus characterization for the 1,015 string pairs used in compilation correctness testing.

Reproduces the exact test corpus using the same LCG and word pool as the Rust implementation,
then analyzes: length distribution, character set coverage, overlap statistics, edge cases.
"""

import json
import os
import math
from collections import Counter
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))

WORDS = [
    "the", "cat", "sat", "on", "mat", "dog", "ran", "in",
    "a", "an", "is", "was", "are", "were", "has", "had",
    "big", "small", "red", "blue", "green", "fast", "slow",
    "hello", "world", "test", "data", "code", "rust", "math",
]

SEEDS = [42, 123, 456, 789, 1337, 2024, 31415, 27182, 99999, 54321]


def lcg_next(state):
    """Same LCG as Rust: state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)"""
    return (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF


def generate_random_pairs(count, seed):
    state = seed
    pairs = []
    for _ in range(count):
        state = lcg_next(state)
        cand_len = int(((state >> 32) % 8) + 1)
        state = lcg_next(state)
        ref_len = int(((state >> 32) % 8) + 1)

        cand_words = []
        for _ in range(cand_len):
            state = lcg_next(state)
            idx = int((state >> 32)) % len(WORDS)
            cand_words.append(WORDS[idx])

        ref_words = []
        for _ in range(ref_len):
            state = lcg_next(state)
            idx = int((state >> 32)) % len(WORDS)
            ref_words.append(WORDS[idx])

        pairs.append((" ".join(cand_words), " ".join(ref_words)))
    return pairs


def standard_test_suite():
    return [
        ("", ""),
        ("hello", ""),
        ("", "hello"),
        ("hello", "hello"),
        ("hello", "world"),
        ("the cat sat", "the cat sat"),
        ("the cat sat", "the dog sat"),
        ("a b c d", "c d e f"),
        ("a b", "a b c d"),
        ("a b c d", "a b"),
        ("the the the", "the cat dog"),
        ("the quick brown fox jumps over the lazy dog",
         "the quick brown fox jumps over the lazy dog"),
        ("the quick brown fox", "the slow brown cat"),
        ("42", "42"),
        ("answer is 42", "the answer is 42"),
    ]


def analyze_corpus():
    # Generate the full corpus for seed=42 (representative)
    standard = standard_test_suite()
    random_pairs = generate_random_pairs(1000, 42)
    all_pairs = standard + random_pairs
    n_pairs = len(all_pairs)

    # Length analysis (in characters)
    cand_char_lens = [len(c) for c, r in all_pairs]
    ref_char_lens = [len(r) for c, r in all_pairs]

    # Length analysis (in tokens/words)
    cand_tok_lens = [len(c.split()) if c else 0 for c, r in all_pairs]
    ref_tok_lens = [len(r.split()) if r else 0 for c, r in all_pairs]

    # Character set analysis
    all_chars = set()
    for c, r in all_pairs:
        all_chars.update(c)
        all_chars.update(r)

    # Token vocabulary
    all_tokens = Counter()
    for c, r in all_pairs:
        if c:
            all_tokens.update(c.split())
        if r:
            all_tokens.update(r.split())

    # Overlap analysis
    exact_matches = sum(1 for c, r in all_pairs if c == r)
    empty_cand = sum(1 for c, r in all_pairs if not c)
    empty_ref = sum(1 for c, r in all_pairs if not r)
    both_empty = sum(1 for c, r in all_pairs if not c and not r)

    # Token overlap per pair
    token_overlaps = []
    for c, r in all_pairs:
        c_toks = set(c.split()) if c else set()
        r_toks = set(r.split()) if r else set()
        if c_toks or r_toks:
            overlap = len(c_toks & r_toks) / max(len(c_toks | r_toks), 1)
            token_overlaps.append(overlap)
        else:
            token_overlaps.append(1.0)

    def percentiles(data):
        s = sorted(data)
        n = len(s)
        return {
            "min": s[0],
            "p5": s[int(n * 0.05)],
            "p25": s[int(n * 0.25)],
            "median": s[int(n * 0.50)],
            "p75": s[int(n * 0.75)],
            "p95": s[int(n * 0.95)],
            "max": s[-1],
            "mean": round(sum(s) / n, 2),
        }

    # Cross-seed consistency check
    cross_seed_sizes = []
    for seed in SEEDS:
        rp = generate_random_pairs(1000, seed)
        sp = standard_test_suite()
        cross_seed_sizes.append(len(sp) + len(rp))

    # Edge case categorization
    edge_categories = {
        "empty_candidate": empty_cand,
        "empty_reference": empty_ref,
        "both_empty": both_empty,
        "exact_match": exact_matches,
        "single_token_pairs": sum(1 for c, r in all_pairs
                                   if len(c.split()) == 1 and len(r.split()) == 1 and c and r),
        "no_token_overlap": sum(1 for c, r in all_pairs
                                if c and r and not (set(c.split()) & set(r.split()))),
        "contains_numbers": sum(1 for c, r in all_pairs
                                if any(ch.isdigit() for ch in c + r)),
    }

    # Length bucket distribution
    def bucket_distribution(lens):
        buckets = {"0": 0, "1-3": 0, "4-8": 0, "9-16": 0, "17-32": 0, "33+": 0}
        for l in lens:
            if l == 0:
                buckets["0"] += 1
            elif l <= 3:
                buckets["1-3"] += 1
            elif l <= 8:
                buckets["4-8"] += 1
            elif l <= 16:
                buckets["9-16"] += 1
            elif l <= 32:
                buckets["17-32"] += 1
            else:
                buckets["33+"] += 1
        return buckets

    results = {
        "meta": {
            "description": "Characterization of the 1,015 test string pairs used in compilation correctness testing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "methodology": "Reproduced exact test corpus using same LCG (6364136223846793005 * state + 1442695040888963407) and word pool as Rust implementation. Analysis on seed=42 (representative); cross-seed consistency verified.",
            "total_pairs": n_pairs,
            "standard_pairs": len(standard),
            "random_pairs": len(random_pairs),
            "generation": "15 hardcoded standard pairs + 1,000 LCG-generated random pairs per seed",
        },
        "character_length_distribution": {
            "candidate": percentiles(cand_char_lens),
            "reference": percentiles(ref_char_lens),
            "candidate_buckets_chars": bucket_distribution(cand_char_lens),
            "reference_buckets_chars": bucket_distribution(ref_char_lens),
        },
        "token_length_distribution": {
            "candidate": percentiles(cand_tok_lens),
            "reference": percentiles(ref_tok_lens),
            "candidate_buckets_tokens": bucket_distribution(cand_tok_lens),
            "reference_buckets_tokens": bucket_distribution(ref_tok_lens),
        },
        "character_set": {
            "unique_characters": len(all_chars),
            "characters": sorted(all_chars),
            "has_unicode": any(ord(c) > 127 for c in all_chars),
            "has_digits": any(c.isdigit() for c in all_chars),
            "has_uppercase": any(c.isupper() for c in all_chars),
            "character_classes": {
                "lowercase_alpha": sum(1 for c in all_chars if c.islower()),
                "uppercase_alpha": sum(1 for c in all_chars if c.isupper()),
                "digits": sum(1 for c in all_chars if c.isdigit()),
                "whitespace": sum(1 for c in all_chars if c.isspace()),
                "other": sum(1 for c in all_chars if not c.isalnum() and not c.isspace()),
            },
        },
        "vocabulary": {
            "unique_tokens": len(all_tokens),
            "total_token_occurrences": sum(all_tokens.values()),
            "top_10_tokens": all_tokens.most_common(10),
            "word_pool_size": len(WORDS),
            "word_pool_coverage": round(len(set(all_tokens.keys()) & set(WORDS)) / len(WORDS), 3),
        },
        "overlap_statistics": {
            "exact_match_pairs": exact_matches,
            "exact_match_fraction": round(exact_matches / n_pairs, 4),
            "token_jaccard_overlap": percentiles([round(x, 4) for x in token_overlaps]),
        },
        "edge_case_coverage": edge_categories,
        "limitations": {
            "narrow_vocabulary": f"Only {len(WORDS)} unique words in random generation pool (all lowercase English, 2-5 chars)",
            "no_unicode": "Random pairs contain only ASCII lowercase + spaces. Standard suite includes digits but no Unicode.",
            "short_strings": f"Max token length is 8 words ({max(max(cand_tok_lens), max(ref_tok_lens))} tokens). Real NLP inputs can be 100+ tokens.",
            "no_adversarial": "No adversarial strings (control characters, very long strings, Unicode edge cases, RTL text)",
            "limited_character_diversity": "All random strings use lowercase a-z + space. No punctuation, casing variation, or special characters.",
        },
        "cross_seed_consistency": {
            "seeds": SEEDS,
            "pairs_per_seed": cross_seed_sizes,
            "all_equal": len(set(cross_seed_sizes)) == 1,
        },
        "recommendations": [
            "Add Unicode test pairs (CJK, Arabic, emoji, combining characters)",
            "Add longer strings (50-200 tokens) to test scaling behavior",
            "Add adversarial pairs (empty tokens, repeated delimiters, control characters)",
            "Add real-world NLP outputs from actual model evaluations",
            "Stratify random generation by length to ensure uniform coverage",
        ],
    }

    out_path = os.path.join(BASE, "corpus_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Corpus analysis written to {out_path}")
    print(f"  {n_pairs} pairs analyzed")
    print(f"  Character lengths: candidate {results['character_length_distribution']['candidate']['mean']} mean, "
          f"reference {results['character_length_distribution']['reference']['mean']} mean")
    print(f"  Token lengths: candidate {results['token_length_distribution']['candidate']['mean']} mean, "
          f"reference {results['token_length_distribution']['reference']['mean']} mean")
    print(f"  Vocabulary: {len(all_tokens)} unique tokens from {len(WORDS)}-word pool")
    print(f"  Edge cases: {sum(edge_categories.values())} categorized instances")
    return results


if __name__ == "__main__":
    analyze_corpus()
