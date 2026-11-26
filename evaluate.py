"""Automated evaluation harness for PawPal+ AI.

Run:  python evaluate.py            (prints a summary, exit code 1 if any case fails)
      python evaluate.py --verbose  (also prints every case)

IMPORTANT: image cases are SYNTHETIC (see pawpal_ai/samples.py) rendered with seeds that were
never used while tuning thresholds. Because the generator and the heuristics were written by
the same author, accuracy here measures internal consistency, NOT real-world accuracy.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from pawpal_ai import ImageInput, StoolAnalysisPipeline
from pawpal_ai.analysis import StoolAnalyzer
from pawpal_ai.guardrails import AIResponseValidator, safety_violations
from pawpal_ai.history import PetHistoryManager
from pawpal_ai.imaging import ImageInputError, ImageProcessor, ImageQualityValidator
from pawpal_ai.knowledge import KnowledgeBase
from pawpal_ai.models import ESCALATION_RANK
from pawpal_ai.prompts import FEW_SHOT
from pawpal_ai.samples import SCORE_KINDS, render_sample, to_bytes
from pawpal_ai.scoring import FecalScoringEngine
from pawpal_system import Owner, Pet

NOW = datetime(2026, 11, 24, 18, 0, 0)
HELD_OUT = 500  # seeds >= 500 were never used for tuning (tuning used 1-8, demo images 101-115)


def at_least(report, level: str) -> bool:
    return ESCALATION_RANK[report.escalation.level] >= ESCALATION_RANK[level]


def flagged_names(report) -> List[str]:
    return [f.name for f in report.observations if f.status == "possible"]


def seeded_pet(species: str, specs: List[Tuple[int, Optional[int], str]]) -> Pet:
    """Pet with seeded history: (days_ago, score, category) tuples."""
    pet = Pet("Eval", species)
    for i, (days, score, cat) in enumerate(specs):
        PetHistoryManager.add(pet, {
            "report_id": f"s{i}", "timestamp": (NOW - timedelta(days=days)).isoformat(timespec="seconds"),
            "image_sha256": "", "species": species.lower(), "score": score, "category": cat, "color": "brown",
            "flags": [], "confidence": 0.6, "escalation": "none"})
    return pet


@dataclass
class Case:
    group: str
    name: str
    run: Callable[[], Tuple[bool, str]]
    guardrail: bool = False


def img_bytes(kind: str, seed: int, **kw) -> bytes:
    return to_bytes(render_sample(kind, seed, **kw))


def analyze(data: bytes, species="Dog", pet: Optional[Pet] = None, symptoms=(), name="eval.png", **kw):
    pipe = kw.pop("pipeline", None) or StoolAnalysisPipeline()
    return pipe.analyze(pet or Pet("Eval", species), ImageInput(data, name), symptoms=symptoms, persist=False, now=NOW)


def build_cases() -> List[Case]:
    cases: List[Case] = []
    add = lambda group, name, fn, guard=False: cases.append(Case(group, name, fn, guard)) or None

    # 1. Normal-looking stool ----------------------------------------------------
    for kind in ("firm_log", "log"):
        for j, bg in enumerate(("tile", "grass", "concrete")):
            def run(kind=kind, bg=bg, j=j):
                r = analyze(img_bytes(kind, HELD_OUT + j, background=bg))
                ok = r.status == "completed" and r.classification.score in (2, 3) and not flagged_names(r) and r.escalation.level == "none"
                return ok, f"score={r.classification.score if r.classification else None} flags={flagged_names(r)} esc={r.escalation.level}"
            add("normal-looking stool", f"{kind}/{bg}", run)

    # 2. Loose stool -------------------------------------------------------------
    for kind in ("piles", "spots"):
        for j, bg in enumerate(("tile", "concrete")):
            def run(kind=kind, bg=bg, j=j):
                r = analyze(img_bytes(kind, HELD_OUT + 10 + j, background=bg))
                ok = r.status == "completed" and r.classification.category == "loose / unformed" and at_least(r, "monitor")
                return ok, f"score={r.classification.score} cat={r.classification.category} esc={r.escalation.level}"
            add("loose stool", f"{kind}/{bg}", run)
    for j, bg in enumerate(("tile", "concrete")):
        def run(bg=bg, j=j):
            r = analyze(img_bytes("puddle", HELD_OUT + 20 + j, background=bg))
            ok = r.status == "completed" and r.classification.score == 7 and at_least(r, "monitor")
            return ok, f"score={r.classification.score} esc={r.escalation.level}"
        add("loose stool", f"puddle/{bg}", run)

    # 3. Very hard stool ----------------------------------------------------------
    for j, bg in enumerate(("tile", "concrete", "grass")):
        def run(bg=bg, j=j):
            r = analyze(img_bytes("pellets", HELD_OUT + 30 + j, background=bg))
            ok = r.status == "completed" and r.classification.category == "very hard / dry" and at_least(r, "monitor")
            return ok, f"score={r.classification.score} cat={r.classification.category} esc={r.escalation.level}"
        add("very hard stool", f"pellets/{bg}", run)

    # 4. Unclear photographs (must be rejected, no fabricated observations) ----------
    unclear = {
        "blurry (a)": lambda: img_bytes("log", HELD_OUT + 40, extra="blur"),
        "blurry (b)": lambda: img_bytes("firm_log", HELD_OUT + 41, extra="blur", background="concrete"),
        "too dark (a)": lambda: img_bytes("log", HELD_OUT + 42, extra="dark"),
        "too dark (b)": lambda: img_bytes("firm_log", HELD_OUT + 43, extra="dark", background="grass"),
        "overexposed": lambda: img_bytes("log", HELD_OUT + 44, extra="overexposed"),
        "blank grey": lambda: img_bytes("blank", 1),
        "empty ground": lambda: img_bytes("empty", HELD_OUT + 45),
        "non-stool blue object": lambda: img_bytes("blue_object", HELD_OUT + 46),
    }
    for name, make in unclear.items():
        def run(make=make):
            r = analyze(make())
            ok = (r.status == "rejected_quality" and not r.observations and r.classification is None
                  and r.escalation.level == "inconclusive")
            return ok, f"status={r.status} issues={r.image_quality.issues if r.image_quality else None}"
        add("unclear photograph", name, run, True)

    # 5. Invalid images -------------------------------------------------------------
    def png(size): return to_bytes(Image.new("RGB", size, (120, 90, 60)))
    gif = io.BytesIO(); Image.new("RGB", (200, 200), (120, 90, 60)).save(gif, format="GIF")
    good = img_bytes("log", HELD_OUT + 50)
    invalid = {
        "empty file": b"",
        "random bytes": os.urandom(2048),
        "plain text named .png": b"hello, this is not an image",
        "tiny 50x50 image": png((50, 50)),
        "unsupported GIF": gif.getvalue(),
        "truncated PNG": good[: len(good) // 3],
    }
    for name, data in invalid.items():
        def run(data=data):
            r = analyze(data)
            ok = r.status == "rejected_input" and not r.observations and r.escalation.level == "inconclusive" and bool(r.input_error)
            return ok, f"status={r.status} error={r.input_error[:60]!r}"
        add("invalid image", name, run, True)
    def unsupported_species():
        r = StoolAnalysisPipeline().analyze(Pet("Hammy", "Hamster"), ImageInput(good), persist=False, now=NOW)
        return r.status == "rejected_input", f"status={r.status}"
    add("invalid image", "unsupported species (hamster)", unsupported_species, True)
    def bad_symptom():
        r = analyze(good, symptoms=("not_a_symptom",))
        return r.status == "rejected_input", f"status={r.status}"
    add("invalid image", "unknown symptom keyword", bad_symptom, True)

    # 6. Concerning discoloration ------------------------------------------------------
    for j in range(3):
        def run(j=j):
            r = analyze(img_bytes("firm_log", HELD_OUT + 60 + j, extra="red_streaks", background=("tile", "concrete", "tile")[j]))
            ok = "red_discoloration" in flagged_names(r) and at_least(r, "contact_vet_soon")
            return ok, f"flags={flagged_names(r)} esc={r.escalation.level}"
        add("concerning discoloration", f"red streaks #{j + 1}", run, True)
    for j in range(2):
        def run(j=j):
            r = analyze(img_bytes("log", HELD_OUT + 70 + j, color="black", background=("tile", "concrete")[j]))
            ok = "dark_or_black_stool" in flagged_names(r) and at_least(r, "contact_vet_soon")
            return ok, f"flags={flagged_names(r)} esc={r.escalation.level}"
        add("concerning discoloration", f"black/tarry-looking #{j + 1}", run, True)
    for j in range(2):
        def run(j=j):
            r = analyze(img_bytes("log", HELD_OUT + 80 + j, color="gray", background="concrete"))
            ok = "pale_or_gray_stool" in flagged_names(r) and at_least(r, "contact_vet_soon")
            return ok, f"flags={flagged_names(r)} esc={r.escalation.level}"
        add("concerning discoloration", f"pale/gray #{j + 1}", run, True)
    for j in range(2):
        def run(j=j):
            r = analyze(img_bytes("log", HELD_OUT + 90 + j, extra="mucus", background="concrete"))
            ok = "mucus_like_coating" in flagged_names(r) and at_least(r, "monitor")
            return ok, f"flags={flagged_names(r)} esc={r.escalation.level}"
        add("concerning discoloration", f"mucus-like coating #{j + 1}", run, True)
    def foreign():
        r = analyze(img_bytes("firm_log", HELD_OUT + 95, extra="foreign"))
        return "foreign_material" in flagged_names(r) and at_least(r, "contact_vet_soon"), f"flags={flagged_names(r)} esc={r.escalation.level}"
    add("concerning discoloration", "foreign inclusions", foreign, True)

    # 7. Missing historical data ------------------------------------------------------
    def no_history():
        r = analyze(img_bytes("log", HELD_OUT + 100))
        return r.comparison.trend == "no_history" and "No earlier" in r.comparison.summary, f"trend={r.comparison.trend}"
    add("missing history", "new pet, no records", no_history)
    def history_other_species_pet():
        r = analyze(img_bytes("log", HELD_OUT + 101), species="Cat")
        return r.comparison.n_previous == 0 and r.classification.score is None, f"n_prev={r.comparison.n_previous}"
    add("missing history", "new cat, no records", history_other_species_pet)

    # 8. Repeated observations ----------------------------------------------------------
    def repeated_loose():
        pet = seeded_pet("Dog", [(5, 6, "loose / unformed"), (3, 5, "loose / unformed"), (1, 6, "loose / unformed")])
        r = analyze(img_bytes("piles", HELD_OUT + 110), pet=pet)
        ok = r.comparison.repeated_abnormal and at_least(r, "contact_vet_soon") and "R11_persistent_abnormal" in r.escalation.rules_triggered
        return ok, f"repeated={r.comparison.repeated_abnormal} esc={r.escalation.level}"
    add("repeated observations", "4th loose stool in 5 days", repeated_loose, True)
    def stable_history():
        pet = seeded_pet("Dog", [(6, 2, "firm, well formed"), (4, 3, "soft but formed"), (2, 2, "firm, well formed")])
        r = analyze(img_bytes("firm_log", HELD_OUT + 111), pet=pet)
        return r.comparison.trend == "stable" and not r.comparison.repeated_abnormal, f"trend={r.comparison.trend}"
    add("repeated observations", "stable normal history", stable_history)
    def looser_trend():
        pet = seeded_pet("Dog", [(6, 2, "firm, well formed"), (4, 2, "firm, well formed"), (2, 2, "firm, well formed")])
        r = analyze(img_bytes("puddle", HELD_OUT + 112), pet=pet)
        return r.comparison.trend == "looser" and r.comparison.score_delta >= 2, f"trend={r.comparison.trend} delta={r.comparison.score_delta}"
    add("repeated observations", "sudden change vs history", looser_trend)
    def duplicate():
        pet = Pet("Eval", "Dog")
        data = img_bytes("log", HELD_OUT + 113)
        pipe = StoolAnalysisPipeline()
        first = pipe.analyze(pet, ImageInput(data), now=NOW)
        second = pipe.analyze(pet, ImageInput(data), now=NOW + timedelta(hours=1))
        ok = first.stored and second.comparison.trend == "duplicate" and not second.stored and len(pet.stool_records) == 1
        return ok, f"first.stored={first.stored} second.trend={second.comparison.trend} second.stored={second.stored}"
    add("repeated observations", "identical photo submitted twice", duplicate)

    # 9. Potentially urgent ----------------------------------------------------------------
    def red_plus_watery():
        r = analyze(img_bytes("puddle", HELD_OUT + 120, extra="red_streaks"))
        return r.escalation.level == "urgent" and "R2_blood_like_plus_loose" in r.escalation.rules_triggered, f"flags={flagged_names(r)} esc={r.escalation.level} cat={r.classification.category}"
    add("potentially urgent", "possible blood + watery stool", red_plus_watery, True)
    def red_plus_symptoms():
        r = analyze(img_bytes("firm_log", HELD_OUT + 121, extra="red_streaks"), symptoms=("lethargy", "pale_gums"))
        return r.escalation.level == "urgent" and "R4_symptoms_plus_stool_change" in r.escalation.rules_triggered, f"esc={r.escalation.level}"
    add("potentially urgent", "possible blood + lethargy/pale gums", red_plus_symptoms, True)
    def normal_plus_vomiting():
        r = analyze(img_bytes("firm_log", HELD_OUT + 122), symptoms=("vomiting",))
        return r.escalation.level == "contact_vet_soon", f"esc={r.escalation.level}"
    add("potentially urgent", "normal stool but vomiting reported", normal_plus_vomiting, True)

    # 10. Cats ------------------------------------------------------------------------------
    def cat_no_score():
        r = analyze(img_bytes("log", HELD_OUT + 130), species="Cat")
        c = r.classification
        return r.status == "completed" and c.score is None and "No numeric scale" in c.scale and bool(c.category), f"score={c.score} cat={c.category}"
    add("cats", "no numeric score for cats", cat_no_score, True)

    # 11. Persistence -------------------------------------------------------------------------
    def persistence():
        owner = Owner("O"); pet = owner.add_pet(Pet("Luna", "Dog"))
        pipe = StoolAnalysisPipeline()
        pipe.analyze(pet, ImageInput(img_bytes("log", HELD_OUT + 140)), now=NOW)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.json"); owner.save_to_json(path); loaded = Owner.load_from_json(path)
        recs = loaded.get_pet("Luna").stool_records
        return len(recs) == 1 and recs[0]["category"] is not None, f"records after reload={len(recs)}"
    add("persistence", "history survives JSON save/load", persistence)

    # 11b. Retrieval (RAG) -------------------------------------------------------------------------
    def cites_blood_source():
        r = analyze(img_bytes("firm_log", HELD_OUT + 150, extra="red_streaks"))
        ids = {i for a in r.associations for i in a.reference_ids} | set(r.escalation.reference_ids)
        return "akc-blood-in-stool" in ids and any(x.ref_id == "akc-blood-in-stool" and x.url.startswith("https://") for x in r.references), f"cited={sorted(ids)}"
    add("retrieval (RAG)", "red-colour report cites the blood-in-stool source", cites_blood_source)
    def cites_diarrhea_source():
        r = analyze(img_bytes("piles", HELD_OUT + 151))
        ids = {i for a in r.associations for i in a.reference_ids}
        return "cornell-diarrhea" in ids, f"cited={sorted(ids)}"
    add("retrieval (RAG)", "loose-stool report cites the diarrhea source", cites_diarrhea_source)
    def scale_reference_for_dog():
        hits = KnowledgeBase().retrieve("fecal score consistency scale", "dog", top_k=1)
        return bool(hits) and hits[0].ref_id == "purina-fecal-score-dog", f"{[h.ref_id for h in hits]}"
    add("retrieval (RAG)", "dog scale query returns the Purina chart (species filter before ranking)", scale_reference_for_dog)
    def cat_gets_cat_scale_and_dog_flag():
        kb = KnowledgeBase()
        scale = kb.retrieve("fecal score consistency scale", "cat", top_k=3)
        blood = kb.retrieve("bright red blood streaks", "cat", top_k=1)
        return (any(h.ref_id == "royalcanin-fecal-score-cat" for h in scale) and bool(blood) and "dog-specific" in blood[0].source), f"{[h.ref_id for h in scale]}"
    add("retrieval (RAG)", "cats see the feline chart; dog-only sources are labelled", cat_gets_cat_scale_and_dog_flag)
    def dog_never_gets_feline_only():
        kb = KnowledgeBase()
        ids = {h.ref_id for q in ("fecal score", "litter", "cat feline", "scale consistency") for h in kb.retrieve(q, "dog", top_k=5)}
        return "royalcanin-fecal-score-cat" not in ids, f"{sorted(ids)}"
    add("retrieval (RAG)", "dogs are never shown the feline-only chart", dog_never_gets_feline_only)
    def kb_missing_degrades_gracefully():
        kb = KnowledgeBase(directory="/nonexistent/path")
        r = analyze(img_bytes("piles", HELD_OUT + 152), pipeline=StoolAnalysisPipeline(knowledge=kb))
        warn = [t for t in r.trace if t.stage == "reference_retrieval" and t.status == "warn"]
        return r.status == "completed" and not r.references and bool(warn) and r.validation.passed, f"trace={[ (t.stage,t.status) for t in r.trace if t.stage=='reference_retrieval']}"
    add("retrieval (RAG)", "missing knowledge base degrades gracefully", kb_missing_degrades_gracefully, True)

    # 12. Medical-safety text filter --------------------------------------------------------------
    unsafe = [
        "Your dog has parvovirus.", "Your cat likely has cancer.", "This is a diagnosis of colitis.",
        "Give 5 mg of probiotic twice a day.", "Stop the heartworm medication immediately.",
        "This blood is definitely caused by an ulcer.", "Colitis is present in this stool.", "Start an antibiotic course and fast your dog for 24 hours.",
    ]
    safe = [
        "Loose stool can be associated with a recent food change.", "Only a veterinarian can establish a diagnosis.",
        "Do not change or stop any medication without veterinary advice.", "This tool cannot diagnose disease.",
        "Reddish areas could be food dye; colitis and many other conditions cannot be told apart in a photo.",
        "Contact a veterinarian within 24-48 hours.",
    ]
    for t in unsafe:
        add("safety text filter", f"blocks: {t[:46]}", (lambda t=t: (bool(safety_violations(t)), str(safety_violations(t)))), True)
    for t in safe:
        add("safety text filter", f"allows: {t[:46]}", (lambda t=t: (not safety_violations(t), str(safety_violations(t)))), True)

    # 13. Constrained model-output (JSON) validation --------------------------------------------------
    v = AIResponseValidator()
    for i, shot in enumerate(FEW_SHOT):
        add("model-output schema", f"few-shot exemplar {i + 1} is valid", (lambda shot=shot: (v.validate_llm_payload(json.loads(shot["answer"]))[0], "")), True)
    bad_payloads = {
        "missing keys": {"image_clear": True},
        "score out of range": {"image_clear": True, "consistency_score": 9, "color": "brown", "shape": "log", "flags": [], "confidence": 0.5, "uncertainty_notes": "x"},
        "unknown flag (e.g. a disease)": {"image_clear": True, "consistency_score": 3, "color": "brown", "shape": "log", "flags": ["parvovirus"], "confidence": 0.5, "uncertainty_notes": "x"},
        "diagnostic language": {"image_clear": True, "consistency_score": 3, "color": "brown", "shape": "log", "flags": [], "confidence": 0.5, "uncertainty_notes": "Your dog has giardia."},
        "empty uncertainty": {"image_clear": True, "consistency_score": 3, "color": "brown", "shape": "log", "flags": [], "confidence": 0.5, "uncertainty_notes": " "},
        "not an object": ["a", "b"],
    }
    for name, payload in bad_payloads.items():
        add("model-output schema", f"rejects: {name}", (lambda p=payload: (not v.validate_llm_payload(p)[0], "")), True)
    return cases


# ---------------------------------------------------------------------------------------------
class AblationBaseline:
    """Same feature extractor but WITHOUT the reliability layer: no quality gate, no safety filter,
    no retrieval, always emits a score. Represents 'AI without specialization'."""

    def __init__(self) -> None:
        self.proc, self.val, self.an, self.scorer = ImageProcessor(), ImageQualityValidator(), StoolAnalyzer(), FecalScoringEngine()

    def run(self, data: bytes):
        loaded = self.proc.load(ImageInput(data))  # raises ImageInputError on bad bytes: counted as unhandled
        q, subject = self.val.assess(loaded)  # computed only to get the mask; q.usable is ignored
        feats, findings = self.an.analyze(subject, q)
        cls = self.scorer.classify(feats, "dog", q)
        flags = [f.name for f in findings if f.status == "possible"]
        return SimpleNamespace(score=cls.score, flags=flags,
                               escalate=bool({"red_discoloration", "dark_or_black_stool", "pale_or_gray_stool"} & set(flags)))


def comparison_study() -> Dict[str, object]:
    base, full = AblationBaseline(), StoolAnalysisPipeline()
    out: Dict[str, object] = {}
    unusable = [img_bytes("log", HELD_OUT + 200 + i, extra=e) for i, e in enumerate(("blur", "dark", "blur", "dark"))]
    unusable += [img_bytes("empty", HELD_OUT + 210), img_bytes("blue_object", HELD_OUT + 211)]
    corrupt = [b"", os.urandom(512)]
    # Baseline: what does it ACTUALLY do with unusable-but-decodable images and with corrupt files?
    base_scored = sum(isinstance(base.run(d).score, int) for d in unusable)
    base_crashed = 0
    for d in corrupt:
        try:
            base.run(d)
        except ImageInputError:
            base_crashed += 1  # the loader rejects it but nothing converts that into a user-facing result
    full_ok = sum(1 for d in unusable + corrupt
                  if (lambda rep: rep.status.startswith("rejected") and not rep.observations and rep.escalation.level == "inconclusive")(
                      full.analyze(Pet("E", "Dog"), ImageInput(d), persist=False, now=NOW)))
    out["unusable_decodable_images_scored_baseline"] = f"{base_scored}/{len(unusable)} (a score was emitted for an image that cannot be assessed)"
    out["corrupt_files_unhandled_exception_baseline"] = f"{base_crashed}/{len(corrupt)}"
    out["unusable_or_corrupt_handled_gracefully_full"] = f"{full_ok}/{len(unusable) + len(corrupt)}"

    # concerning-colour escalation recall and score accuracy on clean images
    concerning = [img_bytes("firm_log", HELD_OUT + 220 + i, extra="red_streaks") for i in range(3)] + \
                 [img_bytes("log", HELD_OUT + 230 + i, color="black") for i in range(3)]
    b_hit = sum(base.run(d).escalate for d in concerning)
    f_hit = sum(at_least(analyze(d), "contact_vet_soon") for d in concerning)
    out["concerning_escalation_recall_baseline"] = f"{b_hit}/{len(concerning)}"
    out["concerning_escalation_recall_full"] = f"{f_hit}/{len(concerning)}"
    out["note"] = ("Baseline has no 'monitor / contact vet / urgent' levels and no symptom or history rules, so it can only say "
                   "escalate-or-not from colour flags. The scoring model itself is shared, so score accuracy is identical by design.")

    # RAG: citation coverage with and without the knowledge base on notable reports
    notable = [img_bytes("piles", HELD_OUT + 240), img_bytes("firm_log", HELD_OUT + 241, extra="red_streaks"),
               img_bytes("log", HELD_OUT + 242, color="black"), img_bytes("log", HELD_OUT + 243, extra="mucus", background="concrete"),
               img_bytes("pellets", HELD_OUT + 244)]
    def coverage(pipe) -> Tuple[int, int]:
        cited = total = 0
        for d in notable:
            r = pipe.analyze(Pet("E", "Dog"), ImageInput(d), persist=False, now=NOW)
            items = [a.reference_ids for a in r.associations if a.basis == "visual_only"] + [r.escalation.reference_ids]
            total += len(items)
            cited += sum(1 for ids in items if ids)
        return cited, total
    c_with, t_with = coverage(StoolAnalysisPipeline())
    c_no, t_no = coverage(StoolAnalysisPipeline(knowledge=KnowledgeBase.empty()))
    out["guidance_items_with_citation_with_RAG"] = f"{c_with}/{t_with}"
    out["guidance_items_with_citation_without_RAG"] = f"{c_no}/{t_no}"
    return out


def consistency_study() -> Dict[str, object]:
    """How often repeated renders of the same condition (different seeds/backgrounds/lighting) get the same class."""
    proc, val, an, scorer = ImageProcessor(), ImageQualityValidator(), StoolAnalyzer(), FecalScoringEngine()
    per_kind, exact, within1, total = {}, 0, 0, 0
    for score, kind in SCORE_KINDS.items():
        preds = []
        for j in range(9):
            bg = ("tile", "grass", "concrete")[j % 3]
            loaded = proc.load(ImageInput(img_bytes(kind, HELD_OUT + 300 + score * 20 + j, background=bg, color=("brown", "dark_brown")[j % 2])))
            q, subject = val.assess(loaded)
            if not q.usable:
                continue
            feats, _ = an.analyze(subject, q)
            preds.append(scorer.classify(feats, "dog", q).score)
        mode_share = Counter(preds).most_common(1)[0][1] / len(preds)
        per_kind[score] = round(mode_share, 2)
        exact += sum(p == score for p in preds)
        within1 += sum(abs(p - score) <= 1 for p in preds)
        total += len(preds)
    return {"modal_class_share_per_kind": per_kind, "mean_modal_share": round(sum(per_kind.values()) / len(per_kind), 2),
            "exact_match_vs_intended_score": f"{exact}/{total} ({exact / total:.0%})", "within_1_of_intended_score": f"{within1}/{total} ({within1 / total:.0%})"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cases = build_cases()
    results = []
    for case in cases:
        try:
            ok, info = case.run()
        except Exception as exc:  # a crash is a failure, never silently ignored
            ok, info = False, f"EXCEPTION {type(exc).__name__}: {exc}"
        results.append((case, ok, info))
        if args.verbose or not ok:
            print(f"[{'PASS' if ok else 'FAIL'}] {case.group:<26} {case.name:<46} {info}")

    total = len(results)
    passed = sum(ok for _c, ok, _i in results)
    by_group: Dict[str, List[bool]] = {}
    for c, ok, _ in results:
        by_group.setdefault(c.group, []).append(ok)
    guard = [ok for c, ok, _ in results if c.guardrail]
    print("\n" + "=" * 74)
    print("PawPal+ AI EVALUATION SUMMARY  (synthetic images; see caveats in evaluate.py / model_card.md)")
    print("=" * 74)
    print(f"{'group':<28}{'passed':>8}{'total':>8}")
    for g, oks in by_group.items():
        print(f"{g:<28}{sum(oks):>8}{len(oks):>8}")
    print("-" * 44)
    print(f"Total tests      : {total}")
    print(f"Passed tests     : {passed}")
    print(f"Failed tests     : {total - passed}")
    print(f"Pass percentage  : {passed / total:.1%}")
    print(f"Guardrail tests  : {sum(guard)}/{len(guard)} passed ({sum(guard) / len(guard):.1%})")

    cons = consistency_study()
    print("\nClassification consistency (9 renders per fecal-score class, held-out seeds):")
    print(f"  mean modal-class share : {cons['mean_modal_share']}   per class: {cons['modal_class_share_per_kind']}")
    print(f"  exact vs intended score: {cons['exact_match_vs_intended_score']}   within +/-1: {cons['within_1_of_intended_score']}")

    comp = comparison_study()
    print("\nBaseline (no reliability layer, no retrieval) vs full system:")
    for k, v in comp.items():
        print(f"  {k}: {v}")
    if args.json:
        print(json.dumps({"total": total, "passed": passed, "consistency": cons, "comparison": comp}, indent=2, default=str))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
