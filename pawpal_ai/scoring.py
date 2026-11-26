"""Fecal scoring: map measured image features to a consistency category and (dogs) a 1-7 score.

Dogs use the Nestle Purina 1-7 descriptions (see knowledge_base/purina_fecal_scoring_dog.md).
Cats get a descriptive category only: no feline numeric scale is applied or invented.

The scorer is a small naive-Bayes-style model whose per-class feature priors are HAND-SET from
the published score descriptions (not fitted to real photographs). Probabilities are tempered
so the model does not sound more certain than it is.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .models import FecalClassification, ImageQualityReport, VisualFeatures

DOG_SCALE = "Nestle Purina fecal score 1-7 (dog); photo-based estimate"
CAT_SCALE = "No numeric scale applied for cats (descriptive category only)"

SCORE_DESCRIPTIONS: Dict[int, str] = {
    1: "Very hard and dry, often individual pellets",
    2: "Firm but not hard, segmented appearance",
    3: "Log-like, little segmentation, moist surface",
    4: "Very moist (soggy), distinct log shape",
    5: "Very moist but with a distinct shape, in piles rather than logs",
    6: "Has texture but no defined shape, piles or spots",
    7: "Watery, no texture, flat puddle",
}

# Species-neutral descriptive category derived from the score band.
COARSE_CATEGORY: Dict[int, str] = {
    1: "very hard / dry",
    2: "firm, well formed",
    3: "soft but formed",
    4: "soft but formed",
    5: "loose / unformed",
    6: "loose / unformed",
    7: "watery / liquid",
}

# feature -> (mean, sigma, weight) per score. Hand-set priors, see module docstring.
_PRIORS: Dict[int, Dict[str, tuple]] = {
    1: {"n_pieces": (4, 2.5, 1.0), "elongation": (0.45, 0.25, 0.8), "edge_roughness": (0.92, 0.12, 0.5),
        "segment_dips": (0.5, 1.5, 0.3), "shine": (0.002, 0.010, 0.8), "coverage": (0.04, 0.03, 0.4), "texture": (12, 6, 0.4)},
    2: {"n_pieces": (1, 1.0, 0.6), "elongation": (0.83, 0.10, 1.0), "edge_roughness": (0.92, 0.12, 0.5),
        "segment_dips": (2.5, 1.5, 1.0), "shine": (0.002, 0.010, 0.8), "coverage": (0.07, 0.05, 0.3), "texture": (9, 6, 0.3)},
    3: {"n_pieces": (1, 1.0, 0.6), "elongation": (0.81, 0.10, 1.0), "edge_roughness": (0.92, 0.12, 0.5),
        "segment_dips": (0.3, 1.0, 1.0), "shine": (0.008, 0.008, 0.8), "coverage": (0.085, 0.05, 0.3), "texture": (7, 6, 0.3)},
    4: {"n_pieces": (1, 1.0, 0.6), "elongation": (0.73, 0.09, 1.0), "edge_roughness": (0.93, 0.12, 0.5),
        "segment_dips": (0.5, 1.5, 0.5), "shine": (0.020, 0.008, 1.0), "coverage": (0.095, 0.05, 0.3), "texture": (8, 6, 0.3)},
    5: {"n_pieces": (2, 1.2, 0.5), "elongation": (0.15, 0.25, 1.0), "edge_roughness": (0.90, 0.10, 0.6),
        "segment_dips": (0.5, 1.5, 0.2), "shine": (0.020, 0.008, 0.8), "coverage": (0.12, 0.06, 0.4), "texture": (8, 6, 0.2)},
    6: {"n_pieces": (2, 1.5, 0.4), "elongation": (0.25, 0.25, 0.7), "edge_roughness": (1.10, 0.12, 1.0),
        "segment_dips": (0.5, 1.5, 0.2), "shine": (0.009, 0.008, 0.6), "coverage": (0.12, 0.06, 0.4), "texture": (8, 6, 0.2)},
    7: {"n_pieces": (1, 1.0, 0.4), "elongation": (0.30, 0.25, 0.6), "edge_roughness": (0.94, 0.12, 0.4),
        "segment_dips": (0.5, 1.5, 0.1), "shine": (0.006, 0.008, 0.5), "coverage": (0.27, 0.09, 1.0), "texture": (6, 5, 0.6)},
}
TEMPERATURE = 3.0  # >1 softens the posterior so it is not over-confident


class FecalScoringEngine:
    """Estimate the consistency class from VisualFeatures."""

    def posterior(self, feats: VisualFeatures) -> Dict[int, float]:
        """Return a tempered probability for each of the seven scores."""
        values = {
            "n_pieces": feats.n_pieces,
            "elongation": feats.elongation,
            "edge_roughness": feats.edge_roughness,
            "segment_dips": feats.segment_dips,
            "shine": feats.shine,
            "coverage": feats.coverage,
            "texture": feats.texture,
        }
        logits: Dict[int, float] = {}
        for score, priors in _PRIORS.items():
            total = 0.0
            for name, (mean, sigma, weight) in priors.items():
                z = (values[name] - mean) / sigma
                total += -0.5 * weight * min(z * z, 16.0)
            logits[score] = total / TEMPERATURE
        peak = max(logits.values())
        exps = {k: math.exp(v - peak) for k, v in logits.items()}
        norm = sum(exps.values())
        return {k: v / norm for k, v in exps.items()}

    def classify(self, feats: VisualFeatures, species: str, quality: ImageQualityReport) -> FecalClassification:
        """Return the classification for `species` ('dog' or 'cat')."""
        post = self.posterior(feats)
        ranked = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
        best, p_best = ranked[0]
        confidence = round(min(0.85, p_best) * (0.55 + 0.45 * quality.score), 3)
        alternatives = [{"score": s, "probability": round(p, 3)} for s, p in ranked[1:3]]
        rationale = (
            f"Estimated from {feats.n_pieces} piece(s), elongation {feats.elongation:.2f}, "
            f"outline roughness {feats.edge_roughness:.2f}, {feats.segment_dips} segmentation band(s), "
            f"glossiness {feats.shine:.3f}. Heuristic estimate, not a trained model."
        )
        if species == "dog":
            return FecalClassification(
                species="dog", scale=DOG_SCALE, score=best, category=COARSE_CATEGORY[best],
                description=SCORE_DESCRIPTIONS[best], confidence=confidence,
                alternatives=alternatives, rationale=rationale,
            )
        # Cats: category only. Alternatives are shown as categories, never as numbers.
        alt_cats: List[Dict[str, object]] = []
        for entry in alternatives:
            cat = COARSE_CATEGORY[entry["score"]]
            if cat != COARSE_CATEGORY[best] and cat not in [a["category"] for a in alt_cats]:
                alt_cats.append({"category": cat, "probability": entry["probability"]})
        coarse_p = sum(p for s, p in post.items() if COARSE_CATEGORY[s] == COARSE_CATEGORY[best])
        confidence = round(min(0.8, coarse_p) * (0.55 + 0.45 * quality.score) * 0.9, 3)
        return FecalClassification(
            species="cat", scale=CAT_SCALE, score=None, category=COARSE_CATEGORY[best],
            description="Descriptive category only; the canine 1-7 scale is not valid for cats.",
            confidence=confidence, alternatives=alt_cats, rationale=rationale,
        )
