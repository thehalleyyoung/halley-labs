#!/usr/bin/env python3
"""
Automated Metamorphic Relation Learner for NLP Pipelines.

Implements specification mining: feeds (input, transformed_input) pairs to
simulated models, observes output changes, clusters them by pattern, and
discovers invariance and directional metamorphic relations automatically.

Usage:
    python3 benchmarks/meta_rule_learner.py
"""

import json
import os
import random
import re
import string
import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MetamorphicRelation:
    """A discovered metamorphic relation."""
    id: str
    transform_name: str
    relation_type: str          # "invariance" | "directional"
    description: str
    precision: float = 0.0
    recall: float = 0.0
    support: int = 0            # number of observations backing the rule
    direction: str | None = None  # for directional: "flip", "shift", "scale", …

@dataclass
class ProbeResult:
    """Single observation from probing a model."""
    original_input: str
    transformed_input: str
    original_output: dict
    transformed_output: dict
    transform_name: str
    output_delta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulated NLP models (stand-ins for real architectures)
# ---------------------------------------------------------------------------

class SimulatedModel:
    """Base class for simulated NLP pipeline models."""

    name: str = "base"

    def predict(self, text: str) -> dict:
        raise NotImplementedError


class T5Model(SimulatedModel):
    """Simulates T5-style encoder-decoder for sentiment + NER."""

    name = "T5-base"

    def predict(self, text: str) -> dict:
        sentiment = self._sentiment(text)
        entities = self._ner(text)
        return {"sentiment": sentiment, "entities": entities, "confidence": self._conf(text)}

    def _sentiment(self, text: str) -> str:
        neg = {"bad", "terrible", "awful", "hate", "worst", "horrible", "poor",
               "not good", "never", "disappointed", "angry", "sad", "fail",
               "ugly", "boring", "mediocre", "disgusting", "dreadful"}
        low = text.lower()
        neg_count = sum(1 for w in neg if w in low)
        if "not bad" in low or "not terrible" in low:
            neg_count = max(0, neg_count - 2)
        return "negative" if neg_count >= 1 else "positive"

    def _ner(self, text: str) -> list[dict]:
        ents = []
        for pat, label in [
            (r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", "PERSON"),
            (r"\b(?:New York|London|Paris|Tokyo|Berlin|Chicago|Boston)\b", "LOC"),
            (r"\b(?:Google|Apple|Microsoft|Amazon|Meta|OpenAI)\b", "ORG"),
        ]:
            for m in re.finditer(pat, text):
                ents.append({"text": m.group(), "label": label, "start": m.start(), "end": m.end()})
        return ents

    @staticmethod
    def _conf(text: str) -> float:
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return 0.70 + (h % 3000) / 10000.0


class LLaMAModel(SimulatedModel):
    """Simulates LLaMA-style autoregressive decoder for classification."""

    name = "LLaMA-7B-sim"

    def predict(self, text: str) -> dict:
        sentiment = self._sentiment(text)
        topics = self._topics(text)
        return {"sentiment": sentiment, "topics": topics, "confidence": self._conf(text)}

    def _sentiment(self, text: str) -> str:
        pos = {"great", "excellent", "wonderful", "love", "best", "amazing",
               "fantastic", "good", "happy", "beautiful", "outstanding", "superb"}
        neg = {"bad", "terrible", "awful", "hate", "worst", "horrible", "poor",
               "never", "disappointed", "angry", "fail", "ugly", "boring"}
        low = text.lower()
        p = sum(1 for w in pos if w in low)
        n = sum(1 for w in neg if w in low)
        if "not" in low.split():
            p, n = n, p
        if p > n:
            return "positive"
        if n > p:
            return "negative"
        return "neutral"

    def _topics(self, text: str) -> list[str]:
        topic_kw = {
            "technology": {"computer", "software", "ai", "algorithm", "model", "data", "gpu", "neural"},
            "sports": {"game", "team", "player", "score", "match", "championship", "coach"},
            "politics": {"government", "election", "president", "vote", "policy", "congress"},
            "science": {"research", "experiment", "hypothesis", "theory", "discovery"},
        }
        words = set(text.lower().split())
        return [t for t, kws in topic_kw.items() if words & kws] or ["general"]

    @staticmethod
    def _conf(text: str) -> float:
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return 0.65 + (h % 3500) / 10000.0


class MultimodalModel(SimulatedModel):
    """Simulates multimodal model operating on text (image features ignored)."""

    name = "CLIP-text-sim"

    def predict(self, text: str) -> dict:
        sentiment = self._sentiment(text)
        embedding = self._embed(text)
        return {"sentiment": sentiment, "embedding_norm": embedding, "confidence": self._conf(text)}

    def _sentiment(self, text: str) -> str:
        score = 0.0
        words = text.lower().split()
        pos = {"great": 2, "good": 1, "excellent": 2, "love": 2, "best": 2,
               "amazing": 2, "wonderful": 2, "happy": 1, "fantastic": 2}
        neg = {"bad": -2, "terrible": -3, "awful": -2, "hate": -2, "worst": -3,
               "horrible": -2, "poor": -1, "disappointing": -2, "boring": -1}
        negate = False
        for w in words:
            if w in ("not", "never", "no", "n't"):
                negate = True
                continue
            s = pos.get(w, 0) + neg.get(w, 0)
            if negate:
                s = -s
                negate = False
            score += s
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    @staticmethod
    def _embed(text: str) -> float:
        h = int(hashlib.md5(text.encode()).hexdigest()[:16], 16)
        return round(math.sqrt(h % 10000) / 100.0, 4)

    @staticmethod
    def _conf(text: str) -> float:
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return 0.60 + (h % 4000) / 10000.0


# ---------------------------------------------------------------------------
# Transformation strategies (perturbation probes)
# ---------------------------------------------------------------------------

CORPUS = [
    "The movie was great and the acting was excellent.",
    "I hate this terrible product, it is the worst.",
    "John Smith visited New York last summer.",
    "Google announced a new AI model yesterday.",
    "The team played an amazing game in the championship.",
    "The government policy disappointed many voters.",
    "This research experiment produced a groundbreaking discovery.",
    "The software update caused a horrible user experience.",
    "Mary Johnson and Bob Williams traveled to London together.",
    "Apple released a fantastic new computer for data science.",
    "The boring lecture never captured the students attention.",
    "An outstanding player scored the winning goal for the team.",
    "Amazon launched a wonderful new algorithm for recommendations.",
    "The president signed an important policy for congress.",
    "Scientists made a beautiful hypothesis about neural theory.",
    "The ugly design of the application made users angry.",
    "Meta developed a superb model for language understanding.",
    "The coach was happy with the match result and score.",
    "OpenAI published excellent research on algorithm safety.",
    "The poor quality of the game left fans disappointed.",
]


def _random_char_perturb(text: str) -> str:
    """Character-level: swap two adjacent characters in a random word."""
    words = text.split()
    if len(words) < 2:
        return text
    idx = random.randint(0, len(words) - 1)
    w = words[idx]
    if len(w) > 2:
        i = random.randint(0, len(w) - 2)
        w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    words[idx] = w
    return " ".join(words)


def _random_word_drop(text: str) -> str:
    """Word-level: drop a random non-key word."""
    words = text.split()
    if len(words) <= 3:
        return text
    drop = random.randint(1, len(words) - 2)
    return " ".join(words[:drop] + words[drop + 1:])


def _synonym_replace(text: str) -> str:
    """Semantic similarity: replace a word with a close synonym."""
    synonyms = {
        "great": "excellent", "excellent": "outstanding", "good": "fine",
        "terrible": "dreadful", "awful": "horrible", "bad": "poor",
        "amazing": "incredible", "wonderful": "marvelous", "happy": "pleased",
        "beautiful": "lovely", "hate": "despise", "love": "adore",
        "boring": "dull", "ugly": "hideous", "fantastic": "superb",
        "horrible": "dreadful", "outstanding": "exceptional",
    }
    words = text.split()
    for i, w in enumerate(words):
        wl = w.lower().strip(string.punctuation)
        if wl in synonyms:
            replacement = synonyms[wl]
            if w[0].isupper():
                replacement = replacement.capitalize()
            trail = w[len(wl):]
            words[i] = replacement + trail
            break
    return " ".join(words)


def _negate_sentiment(text: str) -> str:
    """Negation: insert 'not' before the first sentiment-bearing adjective/verb."""
    targets = {"great", "excellent", "good", "terrible", "awful", "bad",
               "amazing", "wonderful", "happy", "beautiful", "hate", "love",
               "boring", "ugly", "fantastic", "horrible", "outstanding",
               "superb", "poor", "dreadful", "disappointing", "disappointed"}
    words = text.split()
    for i, w in enumerate(words):
        if w.lower().strip(string.punctuation) in targets:
            words.insert(i, "not")
            break
    return " ".join(words)


def _entity_substitute(text: str) -> str:
    """Entity substitution: swap named entities with alternatives."""
    subs = {
        "John Smith": "Alice Brown", "Mary Johnson": "David Lee",
        "Bob Williams": "Carol White",
        "New York": "Chicago", "London": "Berlin", "Paris": "Tokyo",
        "Google": "Microsoft", "Apple": "Amazon", "Meta": "OpenAI",
        "Amazon": "Google", "OpenAI": "Meta", "Microsoft": "Apple",
    }
    result = text
    for orig, repl in subs.items():
        if orig in result:
            result = result.replace(orig, repl, 1)
            break
    return result


def _sentence_paraphrase(text: str) -> str:
    """Sentence-level paraphrase (simple reorder of clauses around 'and')."""
    if " and " in text:
        parts = text.split(" and ", 1)
        return parts[1].strip().capitalize().rstrip(".") + " and " + parts[0].strip().lower() + "."
    return text


TRANSFORMS = {
    "char_perturb": _random_char_perturb,
    "word_drop": _random_word_drop,
    "synonym_replace": _synonym_replace,
    "negate_sentiment": _negate_sentiment,
    "entity_substitute": _entity_substitute,
    "sentence_paraphrase": _sentence_paraphrase,
}


# ---------------------------------------------------------------------------
# Specification miner
# ---------------------------------------------------------------------------

class MetamorphicRelationMiner:
    """
    Mines metamorphic relations by:
      1. Probing a model with (input, transform(input)) pairs
      2. Computing output deltas
      3. Clustering deltas into invariance vs directional patterns
      4. Reporting discovered MRs with precision estimates
    """

    def __init__(self, model: SimulatedModel, corpus: list[str], n_probes: int = 20):
        self.model = model
        self.corpus = corpus
        self.n_probes = n_probes
        self.observations: list[ProbeResult] = []
        self.discovered_mrs: list[MetamorphicRelation] = []

    # -- Phase 1: probing ---------------------------------------------------

    def probe_all(self) -> None:
        """Run all transforms on the corpus and record observations."""
        for transform_name, transform_fn in TRANSFORMS.items():
            for text in self.corpus[:self.n_probes]:
                transformed = transform_fn(text)
                if transformed == text:
                    continue
                orig_out = self.model.predict(text)
                trans_out = self.model.predict(transformed)
                delta = self._compute_delta(orig_out, trans_out)
                self.observations.append(ProbeResult(
                    original_input=text,
                    transformed_input=transformed,
                    original_output=orig_out,
                    transformed_output=trans_out,
                    transform_name=transform_name,
                    output_delta=delta,
                ))

    @staticmethod
    def _compute_delta(orig: dict, trans: dict) -> dict:
        delta: dict[str, Any] = {}
        for k in orig:
            if k not in trans:
                delta[k] = {"removed": orig[k]}
                continue
            if isinstance(orig[k], str) and isinstance(trans[k], str):
                if orig[k] == trans[k]:
                    delta[k] = "unchanged"
                else:
                    delta[k] = {"from": orig[k], "to": trans[k]}
            elif isinstance(orig[k], (int, float)) and isinstance(trans[k], (int, float)):
                diff = trans[k] - orig[k]
                delta[k] = "unchanged" if abs(diff) < 1e-9 else {"shift": round(diff, 6)}
            elif isinstance(orig[k], list) and isinstance(trans[k], list):
                orig_set = {json.dumps(e, sort_keys=True) for e in orig[k]}
                trans_set = {json.dumps(e, sort_keys=True) for e in trans[k]}
                added = trans_set - orig_set
                removed = orig_set - trans_set
                if not added and not removed:
                    delta[k] = "unchanged"
                else:
                    delta[k] = {"added": len(added), "removed": len(removed)}
            else:
                delta[k] = "unchanged" if orig[k] == trans[k] else "changed"
        return delta

    # -- Phase 2: clustering and rule extraction ----------------------------

    def mine_relations(self) -> list[MetamorphicRelation]:
        """Cluster observations per transform and extract MRs."""
        by_transform: dict[str, list[ProbeResult]] = defaultdict(list)
        for obs in self.observations:
            by_transform[obs.transform_name].append(obs)

        mr_id = 0
        for tname, obs_list in by_transform.items():
            if not obs_list:
                continue

            # Check each output field for invariance or directional pattern
            field_patterns: dict[str, Counter] = defaultdict(Counter)
            for obs in obs_list:
                for field_name, dval in obs.output_delta.items():
                    if dval == "unchanged":
                        field_patterns[field_name]["invariant"] += 1
                    elif isinstance(dval, dict) and "from" in dval and "to" in dval:
                        direction = f"{dval['from']}->{dval['to']}"
                        field_patterns[field_name][f"directional:{direction}"] += 1
                    elif isinstance(dval, dict) and "shift" in dval:
                        field_patterns[field_name]["numeric_shift"] += 1
                    elif isinstance(dval, dict) and ("added" in dval or "removed" in dval):
                        field_patterns[field_name]["list_change"] += 1
                    else:
                        field_patterns[field_name]["other_change"] += 1

            total = len(obs_list)
            for field_name, patterns in field_patterns.items():
                most_common_pattern, count = patterns.most_common(1)[0]
                ratio = count / total

                if most_common_pattern == "invariant" and ratio >= 0.75:
                    mr_id += 1
                    self.discovered_mrs.append(MetamorphicRelation(
                        id=f"MR-{mr_id:03d}",
                        transform_name=tname,
                        relation_type="invariance",
                        description=f"{tname} preserves '{field_name}' output "
                                    f"({ratio:.0%} of {total} probes)",
                        precision=ratio,
                        recall=ratio,
                        support=count,
                    ))
                elif most_common_pattern.startswith("directional:") and ratio >= 0.50:
                    direction_str = most_common_pattern.split(":", 1)[1]
                    mr_id += 1
                    self.discovered_mrs.append(MetamorphicRelation(
                        id=f"MR-{mr_id:03d}",
                        transform_name=tname,
                        relation_type="directional",
                        description=f"{tname} causes '{field_name}' to shift "
                                    f"{direction_str} ({ratio:.0%} of {total} probes)",
                        precision=ratio,
                        recall=ratio,
                        support=count,
                        direction=direction_str,
                    ))
                elif most_common_pattern == "list_change" and ratio >= 0.50:
                    mr_id += 1
                    self.discovered_mrs.append(MetamorphicRelation(
                        id=f"MR-{mr_id:03d}",
                        transform_name=tname,
                        relation_type="directional",
                        description=f"{tname} modifies list field '{field_name}' "
                                    f"({ratio:.0%} of {total} probes)",
                        precision=ratio,
                        recall=ratio,
                        support=count,
                        direction="list_modification",
                    ))
                elif most_common_pattern == "numeric_shift" and ratio >= 0.50:
                    mr_id += 1
                    self.discovered_mrs.append(MetamorphicRelation(
                        id=f"MR-{mr_id:03d}",
                        transform_name=tname,
                        relation_type="directional",
                        description=f"{tname} causes numeric shift in '{field_name}' "
                                    f"({ratio:.0%} of {total} probes)",
                        precision=ratio,
                        recall=ratio,
                        support=count,
                        direction="numeric_shift",
                    ))

        return self.discovered_mrs


# ---------------------------------------------------------------------------
# Hand-crafted baseline (the "known" set we compare against)
# ---------------------------------------------------------------------------

HANDCRAFTED_MRS = [
    {"transform": "synonym_replace", "field": "sentiment", "type": "invariance",
     "description": "Synonym replacement preserves sentiment polarity"},
    {"transform": "synonym_replace", "field": "entities", "type": "invariance",
     "description": "Synonym replacement preserves named entities"},
    {"transform": "entity_substitute", "field": "sentiment", "type": "invariance",
     "description": "Entity substitution preserves sentiment"},
    {"transform": "entity_substitute", "field": "entities", "type": "directional",
     "description": "Entity substitution changes entity spans only"},
    {"transform": "negate_sentiment", "field": "sentiment", "type": "directional",
     "description": "Negation flips sentiment polarity"},
    {"transform": "sentence_paraphrase", "field": "sentiment", "type": "invariance",
     "description": "Paraphrase preserves sentiment"},
    {"transform": "char_perturb", "field": "sentiment", "type": "invariance",
     "description": "Minor typos preserve sentiment"},
    {"transform": "word_drop", "field": "topics", "type": "invariance",
     "description": "Dropping a word preserves topic assignment"},
]


def compare_to_handcrafted(discovered: list[MetamorphicRelation]) -> dict:
    """Compare discovered MRs against the hand-crafted baseline."""
    matched = 0
    novel = 0
    handcrafted_keys = set()
    for hc in HANDCRAFTED_MRS:
        handcrafted_keys.add((hc["transform"], hc["field"], hc["type"]))

    discovered_keys = set()
    for mr in discovered:
        field_match = None
        desc_lower = mr.description.lower()
        for candidate in ["sentiment", "entities", "topics", "confidence",
                          "embedding_norm"]:
            if candidate in desc_lower:
                field_match = candidate
                break
        if field_match:
            key = (mr.transform_name, field_match, mr.relation_type)
            discovered_keys.add(key)

    matched = len(handcrafted_keys & discovered_keys)
    novel_keys = discovered_keys - handcrafted_keys
    missed_keys = handcrafted_keys - discovered_keys

    return {
        "handcrafted_total": len(HANDCRAFTED_MRS),
        "discovered_total": len(discovered),
        "matched": matched,
        "novel": len(novel_keys),
        "missed": len(missed_keys),
        "recovery_rate": round(matched / max(len(HANDCRAFTED_MRS), 1), 4),
        "novel_rules": [
            {"transform": k[0], "field": k[1], "type": k[2]} for k in sorted(novel_keys)
        ],
        "missed_rules": [
            {"transform": k[0], "field": k[1], "type": k[2]} for k in sorted(missed_keys)
        ],
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    """Run the full mining experiment across three simulated architectures."""
    models = [T5Model(), LLaMAModel(), MultimodalModel()]
    results: dict[str, Any] = {"models": {}, "aggregate": {}}

    all_discovered: list[MetamorphicRelation] = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Mining metamorphic relations for: {model.name}")
        print(f"{'='*60}")

        miner = MetamorphicRelationMiner(model, CORPUS, n_probes=20)
        miner.probe_all()
        mrs = miner.mine_relations()
        all_discovered.extend(mrs)

        comparison = compare_to_handcrafted(mrs)

        print(f"  Observations collected : {len(miner.observations)}")
        print(f"  MRs discovered         : {len(mrs)}")
        invariance = [m for m in mrs if m.relation_type == "invariance"]
        directional = [m for m in mrs if m.relation_type == "directional"]
        print(f"    Invariance relations : {len(invariance)}")
        print(f"    Directional relations: {len(directional)}")
        avg_prec = sum(m.precision for m in mrs) / max(len(mrs), 1)
        print(f"  Average precision      : {avg_prec:.2%}")
        print(f"  Hand-crafted recovered : {comparison['matched']}/{comparison['handcrafted_total']} "
              f"({comparison['recovery_rate']:.0%})")
        print(f"  Novel rules discovered : {comparison['novel']}")

        print("\n  Discovered MRs:")
        for mr in mrs:
            tag = "INV" if mr.relation_type == "invariance" else "DIR"
            print(f"    [{tag}] {mr.id}: {mr.description}")

        results["models"][model.name] = {
            "observations": len(miner.observations),
            "discovered_mrs": [asdict(m) for m in mrs],
            "num_invariance": len(invariance),
            "num_directional": len(directional),
            "average_precision": round(avg_prec, 4),
            "comparison_to_handcrafted": comparison,
        }

    # Aggregate across all models
    total_discovered = len(all_discovered)
    unique_discovered = set()
    for mr in all_discovered:
        desc_lower = mr.description.lower()
        for candidate in ["sentiment", "entities", "topics", "confidence", "embedding_norm"]:
            if candidate in desc_lower:
                unique_discovered.add((mr.transform_name, candidate, mr.relation_type))
                break

    agg_comparison = compare_to_handcrafted(all_discovered)
    overall_avg_prec = sum(m.precision for m in all_discovered) / max(total_discovered, 1)

    results["aggregate"] = {
        "total_mrs_discovered": total_discovered,
        "unique_mr_patterns": len(unique_discovered),
        "overall_average_precision": round(overall_avg_prec, 4),
        "aggregate_comparison": agg_comparison,
    }

    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"  Total MRs discovered (all models): {total_discovered}")
    print(f"  Unique MR patterns               : {len(unique_discovered)}")
    print(f"  Overall average precision         : {overall_avg_prec:.2%}")
    print(f"  Hand-crafted recovery rate        : {agg_comparison['recovery_rate']:.0%}")
    print(f"  Novel rules beyond hand-crafted   : {agg_comparison['novel']}")

    return results


def main() -> None:
    results = run_experiment()

    out_path = os.path.join(os.path.dirname(__file__), "meta_rule_learner_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to {out_path}")


if __name__ == "__main__":
    main()
