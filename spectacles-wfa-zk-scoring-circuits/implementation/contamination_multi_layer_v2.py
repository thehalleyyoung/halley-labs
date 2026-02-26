#!/usr/bin/env python3
"""
Multi-layer contamination detection experiment.

Tests the three-layer detection pipeline:
1. N-gram overlap (existing PSI module)
2. Embedding similarity (new: character n-gram embeddings + LSH)
3. Distributional divergence (new: Jensen-Shannon divergence)

Evaluates on verbatim, synonym-substitution, and paraphrase contamination
to demonstrate that embedding-based detection catches contamination modes
that pure n-gram methods miss.
"""

import json
import math
import random
import hashlib
from collections import Counter
from typing import List, Dict, Tuple

random.seed(42)

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

CLEAN_QA = [
    ("What is the capital of France?", "Paris"),
    ("What is 2+2?", "4"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("What is the speed of light?", "299,792,458 meters per second"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What year did World War II end?", "1945"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the square root of 144?", "12"),
    ("What is the boiling point of water in Celsius?", "100 degrees"),
    ("What is DNA?", "Deoxyribonucleic acid"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the smallest prime number?", "2"),
    ("What is photosynthesis?", "Process by which plants convert light to energy"),
    ("What is the Pythagorean theorem?", "a squared plus b squared equals c squared"),
]

NOVEL_QA = [
    ("What is the tallest mountain?", "Mount Everest"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is the chemical formula for salt?", "NaCl"),
    ("What is the largest ocean?", "Pacific Ocean"),
    ("Who wrote Pride and Prejudice?", "Jane Austen"),
    ("What is absolute zero?", "Minus 273.15 degrees Celsius"),
    ("What is the speed of sound?", "343 meters per second"),
    ("Who discovered gravity?", "Isaac Newton"),
    ("What is the closest star to Earth?", "The Sun"),
    ("What is pi approximately?", "3.14159"),
]

SYNONYM_MAP = {
    "capital": "main city",
    "wrote": "authored",
    "speed": "velocity",
    "chemical": "molecular",
    "largest": "biggest",
    "painted": "created",
    "square root": "radical",
    "boiling point": "vaporization temperature",
    "discovered": "found",
    "smallest": "littlest",
    "process": "mechanism",
    "plants": "vegetation",
    "convert": "transform",
    "light": "solar radiation",
    "energy": "power",
}

def paraphrase_text(text: str) -> str:
    """Apply synonym substitution."""
    result = text
    for word, replacement in SYNONYM_MAP.items():
        result = result.replace(word, replacement)
    return result

def heavy_paraphrase(question: str, answer: str) -> Tuple[str, str]:
    """Heavy restructuring that changes sentence structure."""
    # Completely rewrite maintaining semantic content
    templates = [
        f"Regarding {question.lower().replace('what is ', '').replace('?', '')}, the answer would be {answer}.",
        f"The response to your query is: {answer}.",
        f"If asked about this topic, one would say {answer}.",
    ]
    return question, random.choice(templates)

# ---------------------------------------------------------------------------
# Detection Layers (pure Python implementations)
# ---------------------------------------------------------------------------

def char_ngram_set(text: str, n: int = 5) -> set:
    text = text.lower()
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()

def ngram_overlap(text1: str, text2: str, n: int = 5) -> float:
    s1 = char_ngram_set(text1, n)
    s2 = char_ngram_set(text2, n)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def char_ngram_embedding(text: str, n: int = 3, dim: int = 256) -> Dict[int, float]:
    """Build character n-gram embedding (sparse vector)."""
    text = text.lower()
    components = Counter()
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        h = int(hashlib.sha256(ngram.encode()).hexdigest(), 16) % dim
        components[h] += 1

    # TF-IDF style: 1 + log(count)
    for k in components:
        components[k] = 1.0 + math.log(components[k])

    # Normalize
    norm = math.sqrt(sum(v*v for v in components.values()))
    if norm > 0:
        for k in components:
            components[k] /= norm

    return dict(components)

def cosine_similarity(emb1: dict, emb2: dict) -> float:
    """Cosine similarity between two sparse embeddings."""
    if not emb1 or not emb2:
        return 0.0
    keys = set(emb1) & set(emb2)
    dot = sum(emb1[k] * emb2[k] for k in keys)
    return max(-1.0, min(1.0, dot))

def token_distribution(text: str) -> Dict[str, float]:
    """Token frequency distribution."""
    tokens = text.lower().split()
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {k: v/total for k, v in counts.items()}

def js_divergence(dist1: dict, dist2: dict) -> float:
    """Jensen-Shannon divergence."""
    all_keys = set(dist1) | set(dist2)
    div = 0.0
    for k in all_keys:
        p = dist1.get(k, 0.0)
        q = dist2.get(k, 0.0)
        m = (p + q) / 2.0
        if p > 0 and m > 0:
            div += p * math.log(p / m)
        if q > 0 and m > 0:
            div += q * math.log(q / m)
    return div / 2.0

def multi_layer_detect(train_texts: List[str], eval_texts: List[str],
                       ngram_threshold=0.5, emb_threshold=0.75, div_threshold=0.70):
    """Multi-layer contamination detection."""
    # Layer 1: N-gram overlap
    max_ngram = 0.0
    for e in eval_texts:
        for t in train_texts:
            overlap = ngram_overlap(t, e)
            max_ngram = max(max_ngram, overlap)

    # Layer 2: Embedding similarity
    train_embs = [char_ngram_embedding(t) for t in train_texts]
    eval_embs = [char_ngram_embedding(e) for e in eval_texts]
    max_sim = 0.0
    for ee in eval_embs:
        for te in train_embs:
            sim = cosine_similarity(ee, te)
            max_sim = max(max_sim, sim)

    # Layer 3: Distributional divergence
    train_combined = " ".join(train_texts)
    eval_combined = " ".join(eval_texts)
    td = token_distribution(train_combined)
    ed = token_distribution(eval_combined)
    js = js_divergence(td, ed)
    contam_score = 1.0 - min(js / math.log(2), 1.0)

    # Results
    ngram_detected = max_ngram > ngram_threshold
    emb_detected = max_sim > emb_threshold
    div_detected = contam_score > div_threshold

    # Use "any layer" mode for detection (most sensitive)
    any_detected = ngram_detected or emb_detected or div_detected
    # Use "majority vote" for high-confidence detection
    majority_detected = sum([ngram_detected, emb_detected, div_detected]) >= 2

    return {
        "ngram_overlap": round(max_ngram, 4),
        "ngram_detected": ngram_detected,
        "embedding_similarity": round(max_sim, 4),
        "embedding_detected": emb_detected,
        "distributional_score": round(contam_score, 4),
        "distributional_detected": div_detected,
        "layers_triggered": sum([ngram_detected, emb_detected, div_detected]),
        "contamination_detected": any_detected,
        "majority_detected": majority_detected,
    }

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiments():
    results = {
        "experiment": "multi_layer_contamination_detection",
        "description": "Three-layer contamination detection: n-gram + embedding + distributional",
        "scenarios": [],
    }

    # Scenario 1: Clean data (no contamination)
    print("Scenario 1: Clean data...")
    train = [q + " " + a for q, a in CLEAN_QA]
    eval_data = [q + " " + a for q, a in NOVEL_QA]
    r = multi_layer_detect(train, eval_data)
    r["scenario"] = "clean"
    r["expected_detection"] = False
    r["correct"] = not r["contamination_detected"]
    results["scenarios"].append(r)
    print(f"  Clean: ngram={r['ngram_overlap']:.3f}, emb={r['embedding_similarity']:.3f}, "
          f"dist={r['distributional_score']:.3f} → detected={r['contamination_detected']}")

    # Scenario 2: Verbatim contamination (100%)
    print("Scenario 2: Verbatim contamination...")
    eval_verbatim = [q + " " + a for q, a in CLEAN_QA[:10]]
    r = multi_layer_detect(train, eval_verbatim)
    r["scenario"] = "verbatim_100pct"
    r["expected_detection"] = True
    r["correct"] = r["contamination_detected"]
    results["scenarios"].append(r)
    print(f"  Verbatim: ngram={r['ngram_overlap']:.3f}, emb={r['embedding_similarity']:.3f}, "
          f"dist={r['distributional_score']:.3f} → detected={r['contamination_detected']}")

    # Scenario 3: Synonym substitution
    print("Scenario 3: Synonym substitution...")
    eval_synonym = [paraphrase_text(q + " " + a) for q, a in CLEAN_QA[:10]]
    r = multi_layer_detect(train, eval_synonym)
    r["scenario"] = "synonym_substitution"
    r["expected_detection"] = True
    r["correct"] = r["contamination_detected"]
    results["scenarios"].append(r)
    print(f"  Synonym: ngram={r['ngram_overlap']:.3f}, emb={r['embedding_similarity']:.3f}, "
          f"dist={r['distributional_score']:.3f} → detected={r['contamination_detected']}")

    # Scenario 4: Heavy paraphrasing
    print("Scenario 4: Heavy paraphrasing...")
    eval_heavy = []
    for q, a in CLEAN_QA[:10]:
        _, new_a = heavy_paraphrase(q, a)
        eval_heavy.append(q + " " + new_a)
    r = multi_layer_detect(train, eval_heavy)
    r["scenario"] = "heavy_paraphrase"
    r["expected_detection"] = True
    r["correct"] = r["contamination_detected"]
    results["scenarios"].append(r)
    print(f"  Heavy para: ngram={r['ngram_overlap']:.3f}, emb={r['embedding_similarity']:.3f}, "
          f"dist={r['distributional_score']:.3f} → detected={r['contamination_detected']}")

    # Scenario 5: Partial contamination (20%)
    print("Scenario 5: Partial contamination (20%)...")
    eval_partial = [q + " " + a for q, a in CLEAN_QA[:3]] + \
                   [q + " " + a for q, a in NOVEL_QA[:7]]
    r = multi_layer_detect(train, eval_partial)
    r["scenario"] = "partial_20pct"
    r["expected_detection"] = True
    r["correct"] = r["contamination_detected"]
    results["scenarios"].append(r)
    print(f"  Partial 20%: ngram={r['ngram_overlap']:.3f}, emb={r['embedding_similarity']:.3f}, "
          f"dist={r['distributional_score']:.3f} → detected={r['contamination_detected']}")

    # Scenario 6: Word shuffling + synonym (adversarial)
    print("Scenario 6: Adversarial (shuffle + synonym)...")
    eval_adversarial = []
    for q, a in CLEAN_QA[:10]:
        text = paraphrase_text(q + " " + a)
        words = text.split()
        random.shuffle(words)
        eval_adversarial.append(" ".join(words))
    r = multi_layer_detect(train, eval_adversarial)
    r["scenario"] = "adversarial_shuffle_synonym"
    r["expected_detection"] = True
    r["correct"] = r["contamination_detected"]
    results["scenarios"].append(r)
    print(f"  Adversarial: ngram={r['ngram_overlap']:.3f}, emb={r['embedding_similarity']:.3f}, "
          f"dist={r['distributional_score']:.3f} → detected={r['contamination_detected']}")

    # Summary
    correct = sum(1 for s in results["scenarios"] if s["correct"])
    total = len(results["scenarios"])

    # Layer detection comparison
    ngram_only_correct = 0
    multi_correct = 0
    for s in results["scenarios"]:
        expected = s["expected_detection"]
        ngram_only = s["ngram_detected"]
        multi = s["contamination_detected"]
        if ngram_only == expected:
            ngram_only_correct += 1
        if multi == expected:
            multi_correct += 1

    results["summary"] = {
        "total_scenarios": total,
        "correct_detections": correct,
        "accuracy": round(correct / total, 3),
        "ngram_only_accuracy": round(ngram_only_correct / total, 3),
        "multi_layer_accuracy": round(multi_correct / total, 3),
        "improvement_over_ngram": round((multi_correct - ngram_only_correct) / total, 3),
    }

    print(f"\n{'='*60}")
    print(f"Multi-layer accuracy: {results['summary']['multi_layer_accuracy']:.1%}")
    print(f"N-gram only accuracy: {results['summary']['ngram_only_accuracy']:.1%}")
    print(f"Improvement: +{results['summary']['improvement_over_ngram']:.1%}")

    return results


if __name__ == "__main__":
    results = run_experiments()
    with open("contamination_multi_layer_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to contamination_multi_layer_v2.json")
