"""Reliability and safety: output validation, medical-language guardrails, confidence and stability."""

from __future__ import annotations

import io
import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .models import (ESCALATION_LEVELS, FINDING_STATUSES, AnalysisReport, ConfidenceSummary, Escalation,
                     ValidationResult)

DISCLAIMERS = [
    "PawPal+ AI is a monitoring aid, not a veterinary diagnostic tool. It does not diagnose disease.",
    "Parasites, bacteria, viruses and microscopic blood cannot be reliably identified from a stool photograph.",
    "Only a qualified veterinarian can establish a diagnosis or advise on treatment or medication.",
]
UNCERTAINTY_NOTE = (
    "Observations come from a heuristic image-feature prototype that has not been validated on real photographs; "
    "treat them as estimates."
)

# --- text-safety rules ------------------------------------------------------ #
_HEDGES = re.compile(r"\b(can be|could be|may|might|possible|possibly|could|associated|described|cannot|can't|can not|unclear|appear|appears|if )\b", re.I)
_NEGATIONS = re.compile(r"\b(not|no|cannot|can't|never|without|only a|isn't|does not|do not|don't)\b", re.I)
_DISEASES = re.compile(
    r"\b(parvo(virus)?|giardia(sis)?|coccidia|whipworm|hookworm|roundworm|tapeworm|lymphoma|cancer|tumou?r|"
    r"pancreatitis|colitis|ibd|inflammatory bowel|addison|ulcer|poison(ing)?|toxicity|salmonella|e\. coli|"
    r"hemorrhagic gastroenteritis|megacolon|liver (disease|failure)|kidney (disease|failure))\b", re.I)
_CERTAINTY = re.compile(r"\b(definitely|certainly|guarantee[d]?|100\s?%|proves?|proven|confirmed (that|the)|is caused by|was caused by|caused by)\b", re.I)
_DIAGNOSIS_CLAIM = re.compile(
    r"\b(your|the)\s+(dog|cat|pet|puppy|kitten)\s+(has|have|is suffering from|suffers from|is diagnosed with|likely has|probably has)\b", re.I)
_DIAGNOSIS_WORD = re.compile(r"\bdiagnos(is|ed|e|es|ing)\b", re.I)
_MEDICATION_DIRECTIVE = re.compile(
    r"\b(give|giving|administer|start|begin|stop|discontinue|increase|decrease|reduce|skip|double|withhold|fast)\b"
    r"[^.]{0,50}\b(medication|medicine|dose|dosage|antibiotic|antibiotics|probiotic|probiotics|pill|pills|drug|drugs|"
    r"dewormer|steroid|painkiller|food|meal|meals)\b", re.I)
_DOSAGE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|iu|tablets?|capsules?)\b", re.I)
_MED_SAFE_CONTEXT = re.compile(r"\b(do not|don't|never|without veterinary|without a veterinarian|ask (a|your) veterinarian|until a veterinarian advises)\b", re.I)


def split_sentences(text: str) -> List[str]:
    """Rough sentence splitter good enough for rule matching."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def safety_violations(text: str) -> List[str]:
    """Return rule names violated by `text` (empty list = safe)."""
    problems: List[str] = []
    for sentence in split_sentences(text):
        if _DIAGNOSIS_CLAIM.search(sentence):
            problems.append("diagnosis_claim")
        if _DIAGNOSIS_WORD.search(sentence) and not _NEGATIONS.search(sentence):
            problems.append("unqualified_diagnosis_word")
        if _DISEASES.search(sentence) and not _HEDGES.search(sentence):
            problems.append("unhedged_disease_mention")
        if _CERTAINTY.search(sentence):
            problems.append("overstated_certainty_or_causation")
        if _MEDICATION_DIRECTIVE.search(sentence) and not _MED_SAFE_CONTEXT.search(sentence):
            problems.append("medication_or_feeding_directive")
        if _DOSAGE.search(sentence):
            problems.append("dosage_specified")
    return sorted(set(problems))


class AIResponseValidator:
    """Validates report structure and scrubs unsafe medical language before anything is shown."""

    # ----- structure -------------------------------------------------------- #
    def structural_issues(self, r: AnalysisReport) -> List[str]:
        """Return structural/consistency problems in `r` (empty = well formed)."""
        issues: List[str] = []
        if r.status not in ("completed", "rejected_input", "rejected_quality"):
            issues.append(f"invalid status {r.status!r}")
        if r.escalation is None:
            issues.append("missing escalation")
        elif r.escalation.level not in ESCALATION_LEVELS:
            issues.append(f"invalid escalation level {r.escalation.level!r}")
        if r.status != "completed":
            if r.classification is not None or r.observations:
                issues.append("rejected analysis must not contain classification or observations")
            if r.escalation and r.escalation.level != "inconclusive":
                issues.append("rejected analysis must be 'inconclusive'")
            return issues
        for name in ("image_quality", "classification", "confidence", "comparison"):
            if getattr(r, name) is None:
                issues.append(f"missing {name}")
        if not r.observations:
            issues.append("missing observations")
        if not r.next_steps:
            issues.append("missing next_steps")
        if r.image_quality and not r.image_quality.usable:
            issues.append("completed analysis on an unusable image")
        c = r.classification
        if c:
            if c.species == "dog" and (c.score is None or not 1 <= c.score <= 7):
                issues.append("dog score must be an integer 1-7")
            if c.species == "cat" and c.score is not None:
                issues.append("cats must not receive a numeric score")
            if not 0.0 <= c.confidence <= 1.0:
                issues.append("classification confidence outside 0-1")
        if r.confidence and not 0.0 <= r.confidence.overall <= 1.0:
            issues.append("overall confidence outside 0-1")
        for f in r.observations:
            if f.status not in FINDING_STATUSES:
                issues.append(f"finding {f.name} has invalid status")
            if not 0.0 <= f.confidence <= 1.0:
                issues.append(f"finding {f.name} confidence outside 0-1")
            if f.confidence > 0.95:
                issues.append(f"finding {f.name} claims implausibly high confidence")
        for a in r.associations:
            if a.basis not in ("visual_only", "history_context"):
                issues.append("association has invalid basis")
        return issues

    # ----- text safety ------------------------------------------------------ #
    def _texts(self, r: AnalysisReport) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for i, a in enumerate(r.associations):
            items.append((f"associations[{i}]", a.statement))
        for i, s in enumerate(r.next_steps):
            items.append((f"next_steps[{i}]", s))
        if r.escalation:
            items.append(("escalation.headline", r.escalation.headline))
            for i, s in enumerate(r.escalation.reasons):
                items.append((f"escalation.reasons[{i}]", s))
        for i, f in enumerate(r.observations):
            items.append((f"observations[{i}]", f.detail))
        if r.comparison:
            items.append(("comparison.summary", r.comparison.summary))
        return items

    def enforce(self, r: AnalysisReport) -> ValidationResult:
        """Remove unsafe sentences/items in place, add mandatory disclaimers, and report what happened."""
        redactions: List[str] = []
        kept_assoc = []
        for i, a in enumerate(r.associations):
            bad = safety_violations(a.statement)
            if bad:
                redactions.append(f"associations[{i}] removed: {', '.join(bad)}")
            else:
                kept_assoc.append(a)
        r.associations = kept_assoc
        kept_steps = []
        for i, s in enumerate(r.next_steps):
            bad = safety_violations(s)
            if bad:
                redactions.append(f"next_steps[{i}] removed: {', '.join(bad)}")
            else:
                kept_steps.append(s)
        r.next_steps = kept_steps
        for f in r.observations:
            bad = safety_violations(f.detail)
            if bad:
                redactions.append(f"observation {f.name} detail replaced: {', '.join(bad)}")
                f.detail = "(detail withheld by safety filter)"
        if r.escalation:
            bad = safety_violations(r.escalation.headline) + [x for s in r.escalation.reasons for x in safety_violations(s)]
            if bad:
                redactions.append("escalation text replaced: " + ", ".join(sorted(set(bad))))
                r.escalation = Escalation(r.escalation.level, "Please consult a veterinarian about this result.",
                                          ["Details withheld by safety filter."], r.escalation.rules_triggered,
                                          r.escalation.reference_ids)
        if r.comparison:
            bad = safety_violations(r.comparison.summary)
            if bad:
                redactions.append("comparison summary replaced: " + ", ".join(bad))
                r.comparison.summary = "Comparison text withheld by safety filter."

        # Mandatory content.
        if r.status == "completed" and not r.next_steps:
            r.next_steps = ["Share this report with your veterinarian; it cannot replace an examination."]
        r.disclaimers = list(DISCLAIMERS)
        if r.status == "completed" and r.confidence is not None and UNCERTAINTY_NOTE not in r.confidence.caveats:
            r.confidence.caveats.append(UNCERTAINTY_NOTE)

        issues = self.structural_issues(r)
        remaining = [f"{where}: {', '.join(safety_violations(text))}" for where, text in self._texts(r) if safety_violations(text)]
        issues.extend(remaining)
        return ValidationResult(passed=not issues, issues=issues, redactions=redactions)

    # ----- optional LLM payloads ------------------------------------------- #
    LLM_FLAGS = ("red_discoloration", "dark_or_black_stool", "pale_or_gray_stool", "yellow_orange_stool",
                 "mucus_like_coating", "foreign_material")

    def validate_llm_payload(self, payload: Any) -> Tuple[bool, List[str]]:
        """Check that a multimodal-model JSON answer matches the constrained schema and is safe."""
        issues: List[str] = []
        if not isinstance(payload, dict):
            return False, ["payload is not a JSON object"]
        required = {"image_clear": bool, "consistency_score": (int, type(None)), "color": str, "shape": str,
                    "flags": list, "confidence": (int, float), "uncertainty_notes": str}
        for key, typ in required.items():
            if key not in payload:
                issues.append(f"missing key {key!r}")
            elif not isinstance(payload[key], typ) or (typ is bool and not isinstance(payload[key], bool)):
                issues.append(f"key {key!r} has wrong type")
        if issues:
            return False, issues
        if payload["consistency_score"] is not None and not 1 <= payload["consistency_score"] <= 7:
            issues.append("consistency_score outside 1-7")
        if not 0 <= payload["confidence"] <= 1:
            issues.append("confidence outside 0-1")
        unknown = [f for f in payload["flags"] if f not in self.LLM_FLAGS]
        if unknown:
            issues.append(f"unknown flags {unknown}")
        for key in ("color", "shape", "uncertainty_notes"):
            bad = safety_violations(str(payload[key]))
            if bad:
                issues.append(f"{key} violates safety rules: {', '.join(bad)}")
        if not payload["uncertainty_notes"].strip():
            issues.append("uncertainty_notes must not be empty")
        return not issues, issues


# --------------------------------------------------------------------------- #
class ReliabilityEvaluator:
    """Combines quality, classification confidence and stability under small perturbations."""

    PERTURBATIONS = ("brightness_down", "brightness_up", "jpeg_q60", "downscale_85")

    def perturb(self, rgb: np.ndarray, name: str) -> np.ndarray:
        """Return a slightly altered copy of `rgb` (the kind of variation a real camera adds)."""
        if name == "brightness_down":
            return np.clip(rgb.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        if name == "brightness_up":
            return np.clip(rgb.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        if name == "jpeg_q60":
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=60)
            return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)
        if name == "downscale_85":
            img = Image.fromarray(rgb)
            small = img.resize((max(8, int(img.size[0] * 0.85)), max(8, int(img.size[1] * 0.85))), Image.BILINEAR)
            return np.asarray(small, dtype=np.uint8)
        raise ValueError(name)

    def stability(self, loaded, validator, analyzer, scorer, species: str, base_score, base_category: str) -> Dict[str, Any]:
        """Re-run assessment on perturbed copies; report the fraction that agree with the base result."""
        agree, details = 0, []
        for name in self.PERTURBATIONS:
            variant = replace(loaded, rgb=self.perturb(loaded.rgb, name))
            q, subject = validator.assess(variant)
            if not q.usable:
                details.append({"perturbation": name, "usable": False})
                continue
            feats, _ = analyzer.analyze(subject, q)
            cls = scorer.classify(feats, species, q)
            same = (cls.score == base_score) if species == "dog" else (cls.category == base_category)
            agree += int(same)
            details.append({"perturbation": name, "score": cls.score, "category": cls.category, "agrees": same})
        return {"agreement": round(agree / len(self.PERTURBATIONS), 2), "variants": details}

    def summarise(self, quality, classification, subject_conf: float, stability: Optional[Dict[str, Any]],
                  llm_agreement: Optional[float], dim_light: bool) -> ConfidenceSummary:
        """Blend the components into one overall confidence and explain the limits."""
        comps = {
            "image_quality": round(quality.score, 2),
            "subject_visibility": round(subject_conf, 2),
            "classification": round(classification.confidence, 2),
        }
        weights = {"image_quality": 0.2, "subject_visibility": 0.1, "classification": 0.5}
        if stability is not None:
            comps["stability_under_perturbation"] = stability["agreement"]
            weights["stability_under_perturbation"] = 0.2
        if llm_agreement is not None:
            comps["second_opinion_agreement"] = round(llm_agreement, 2)
            weights["second_opinion_agreement"] = 0.2
        total = sum(weights.values())
        overall = sum(comps[k] * weights[k] for k in comps) / total
        overall = round(min(overall, 0.9), 3)
        level = "high" if overall >= 0.70 else "moderate" if overall >= 0.50 else "low"
        caveats: List[str] = []
        if classification.confidence < 0.35:
            caveats.append("The consistency estimate is borderline between neighbouring categories (see alternatives).")
        if stability is not None and stability["agreement"] < 0.75:
            caveats.append("The result changed under small image perturbations, so it is not stable.")
        if dim_light:
            caveats.append("Dim lighting makes colour judgements unreliable.")
        if quality.warnings:
            caveats.append("Photo-quality warnings: " + ", ".join(quality.warnings) + ".")
        if llm_agreement is not None and llm_agreement < 0.5:
            caveats.append("The optional second opinion disagreed with the heuristic analysis.")
        return ConfidenceSummary(overall=overall, level=level, components=comps, caveats=caveats)
