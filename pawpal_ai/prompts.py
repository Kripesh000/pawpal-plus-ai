"""Structured prompting for the OPTIONAL multimodal second-opinion backend.

The offline heuristic pipeline does not use prompts. These are used only when a multimodal
model is configured (see llm_backend.py). They combine: domain instructions, a constrained
JSON output schema, retrieved reference passages, and few-shot examples.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

from .models import RetrievedReference

SYSTEM_PROMPT = """You are a careful visual-observation assistant inside a pet-care monitoring app.
You are NOT a veterinarian and must never diagnose.

Rules:
1. Describe only what is visible. Separate observations from possible associations.
2. Never name a disease as the likely cause of anything you see. Never claim causation.
3. Never recommend starting, stopping, or changing any medication, diet, fasting, or dose.
4. If the image is blurry, dark, shows no stool, or you are unsure, set image_clear=false, consistency_score=null, and explain in uncertainty_notes. Do not guess.
5. For dogs, estimate the Purina 1-7 fecal score (1 very hard ... 7 watery). For cats, set consistency_score to null; no numeric cat scale is used here.
6. Possible visual flags are limited to: red_discoloration, dark_or_black_stool, pale_or_gray_stool, yellow_orange_stool, mucus_like_coating, foreign_material.
7. Parasites, bacteria and microscopic blood cannot be seen in photos; do not claim to see them.
8. Answer with ONE JSON object and nothing else."""

OUTPUT_SCHEMA = {
    "image_clear": "boolean",
    "consistency_score": "integer 1-7 for dogs, or null",
    "color": "short colour description",
    "shape": "short shape description",
    "flags": "list drawn only from the allowed flag names",
    "confidence": "number 0-1",
    "uncertainty_notes": "non-empty sentence stating what you are unsure about",
}

# Illustrative text-only exemplars written for this project (not real model outputs).
FEW_SHOT: List[Dict[str, str]] = [
    {
        "situation": "Dog. Firm brown log-shaped stool with visible cracks on a white tile, good light.",
        "answer": json.dumps({"image_clear": True, "consistency_score": 2, "color": "medium brown", "shape": "single segmented log",
                              "flags": [], "confidence": 0.7,
                              "uncertainty_notes": "Score is an estimate from one photo; texture detail is limited by resolution."}),
    },
    {
        "situation": "Dog. Brown stool with a thin bright-red streak on one side, loose piles, daylight.",
        "answer": json.dumps({"image_clear": True, "consistency_score": 6, "color": "brown with a red streak", "shape": "irregular piles",
                              "flags": ["red_discoloration"], "confidence": 0.55,
                              "uncertainty_notes": "The red area might be food dye or a reflection; a photo cannot show whether it is blood."}),
    },
    {
        "situation": "Very dark, out-of-focus photo where no stool can be identified.",
        "answer": json.dumps({"image_clear": False, "consistency_score": None, "color": "not assessable", "shape": "not assessable",
                              "flags": [], "confidence": 0.0,
                              "uncertainty_notes": "The image is too dark and blurry to describe any stool."}),
    },
]


def build_user_prompt(species: str, references: Sequence[RetrievedReference], history_summary: str) -> str:
    """Compose the per-request prompt: context, retrieved references, history, schema, exemplars."""
    refs = "\n".join(f"- [{r.ref_id}] {r.title}: {r.snippet}" for r in references) or "- (no references retrieved)"
    shots = "\n\n".join(f"Example situation: {s['situation']}\nExample answer: {s['answer']}" for s in FEW_SHOT)
    return (
        f"Species: {species}\n"
        f"Reference passages (cite by id if relevant; do not invent others):\n{refs}\n\n"
        f"Pet history summary (context only): {history_summary}\n\n"
        f"Required JSON schema: {json.dumps(OUTPUT_SCHEMA)}\n\n{shots}\n\n"
        "Now analyse the attached photo and answer with the JSON object only."
    )
