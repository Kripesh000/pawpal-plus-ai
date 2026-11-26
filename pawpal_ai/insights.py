"""Diet/medication context, possible associations, next steps and veterinary escalation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

from pawpal_system import Pet

from .history import is_abnormal_consistency
from .models import (ESCALATION_RANK, Association, Escalation, FecalClassification, Finding,
                     HistoricalComparison)

SYMPTOMS = ("lethargy", "vomiting", "not_eating", "not_drinking", "pale_gums", "straining", "abdominal_pain")
STANDING_ADVICE = (
    "Regardless of this analysis: if your pet is lethargic, vomiting repeatedly, refusing food or water, has pale gums, "
    "seems to be in pain, or you are simply worried, contact a veterinarian."
)
LOOSE_CATEGORIES = ("loose / unformed", "watery / liquid")


def flagged(findings: Sequence[Finding]) -> Dict[str, Finding]:
    """Findings with status 'possible', keyed by name."""
    return {f.name: f for f in findings if f.status == "possible"}


# --------------------------------------------------------------------------- #
class DietCorrelationEngine:
    """Looks at the pet's Feeding and Medication tasks around the observation date.

    It reports *what was scheduled* and flags changes. It never states that diet or a
    medication caused anything.
    """

    def __init__(self, window_days: int = 7, baseline_days: int = 14) -> None:
        self.window_days = window_days
        self.baseline_days = baseline_days

    def context(self, pet: Pet, now: datetime) -> Dict[str, object]:
        """Return structured diet/medication context for the window ending at `now`."""
        today = now.date()
        start = today - timedelta(days=self.window_days)
        base_start = today - timedelta(days=self.baseline_days)

        def in_range(task, lo, hi):
            d = datetime.strptime(task.date, "%Y-%m-%d").date()
            return lo <= d <= hi

        feeding = [t for t in pet.tasks if t.category == "Feeding"]
        meds = [t for t in pet.tasks if t.category == "Medication"]
        recent_feed = [t for t in feeding if in_range(t, start, today)]
        older_feed = [t for t in feeding if in_range(t, base_start, start - timedelta(days=1))]
        original = {t.description.lower(): t.description for t in recent_feed}
        older_names = {t.description.lower() for t in older_feed}
        new_entries = sorted(original[k] for k in set(original) - older_names) if older_names else []
        recent_per_day = len(recent_feed) / max(self.window_days, 1)
        older_per_day = len(older_feed) / max(self.baseline_days - self.window_days, 1) if older_feed else None
        recent_meds = sorted({t.description for t in meds if in_range(t, start, today)})
        return {
            "feeding_entries_recent": len(recent_feed),
            "new_feeding_entries": new_entries,
            "feeding_per_day_recent": round(recent_per_day, 2),
            "feeding_per_day_baseline": round(older_per_day, 2) if older_per_day is not None else None,
            "medications_recent": recent_meds,
        }

    def associations(self, pet: Pet, now: datetime, relevant: bool, refs: Dict[str, List[str]]) -> List[Association]:
        """Possible diet/medication associations; only produced when the observation is notable."""
        if not relevant:
            return []
        ctx = self.context(pet, now)
        out: List[Association] = []
        if ctx["new_feeding_entries"]:
            names = ", ".join(f"'{n}'" for n in ctx["new_feeding_entries"])
            out.append(Association(
                f"New feeding entries appeared in the last {self.window_days} days ({names}). Recent food or treat changes can be "
                "associated with stool changes; this record alone cannot show that they are related.",
                "history_context", reference_ids=refs.get("diet", [])))
        base = ctx["feeding_per_day_baseline"]
        if base is not None and abs(ctx["feeding_per_day_recent"] - base) >= 0.5:
            out.append(Association(
                f"The scheduled feeding frequency changed (about {ctx['feeding_per_day_recent']} per day recently vs {base} before). "
                "Changes in feeding routine can be associated with changes in stool.",
                "history_context", reference_ids=refs.get("diet", [])))
        if ctx["medications_recent"]:
            names = ", ".join(f"'{n}'" for n in ctx["medications_recent"])
            out.append(Association(
                f"Medication entries were scheduled in the last {self.window_days} days ({names}). Some medications can be associated "
                "with stool changes. Do not change or stop any medication without veterinary advice.",
                "history_context", reference_ids=[]))
        return out


# --------------------------------------------------------------------------- #
class HealthInsightGenerator:
    """Wording for associations and next steps, always hedged and always sourced where possible."""

    def associations(self, cls: FecalClassification, findings: Sequence[Finding], refs: Dict[str, List[str]]) -> List[Association]:
        """Possible associations for the visual observations (never diagnoses)."""
        out: List[Association] = []
        flags = flagged(findings)
        cat = cls.category
        if cat == "very hard / dry":
            out.append(Association(
                "Very hard, dry stool can be associated with low water intake or constipation. A veterinarian can tell whether this applies.",
                "visual_only", reference_ids=refs.get("hard", [])))
        if cat in LOOSE_CATEGORIES:
            out.append(Association(
                "Loose or watery stool can be associated with a recent food change, eating something unusual, stress, or intestinal "
                "irritation or infection. A photograph cannot tell these apart.",
                "visual_only", reference_ids=refs.get("loose", [])))
        if "red_discoloration" in flags:
            out.append(Association(
                "Reddish areas can be associated with fresh bleeding from the lower digestive tract, although food dyes can look similar.",
                "visual_only", reference_ids=refs.get("red", [])))
        if "dark_or_black_stool" in flags:
            out.append(Association(
                "Very dark or black stool can be associated with digested blood from higher in the digestive tract, but some diets "
                "and supplements also darken stool.",
                "visual_only", reference_ids=refs.get("black", [])))
        if "pale_or_gray_stool" in flags:
            out.append(Association(
                "Pale gray or whitish stool is described by a consumer source as possibly related to incomplete digestion; the cause cannot be seen in a photo.",
                "visual_only", reference_ids=refs.get("pale", [])))
        if "yellow_orange_stool" in flags:
            out.append(Association(
                "Yellow-orange colour is described by a consumer source as possibly related to the liver or bile system, and diet colourants can also cause it.",
                "visual_only", reference_ids=refs.get("yellow", [])))
        if "mucus_like_coating" in flags:
            out.append(Association(
                "A glossy coating that could be mucus can be associated with irritation of the colon, a recent diet change, parasites or stress; reflections can look similar.",
                "visual_only", reference_ids=refs.get("mucus", [])))
        if "foreign_material" in flags:
            out.append(Association(
                "Differently coloured pieces could be swallowed non-food items, or could be a lighting artefact.",
                "visual_only", reference_ids=refs.get("foreign", [])))
        return out

    def next_steps(self, cls: FecalClassification, findings: Sequence[Finding], comparison: HistoricalComparison,
                   escalation: Escalation, quality_warnings: Sequence[str]) -> List[str]:
        """Practical, non-medical next steps."""
        steps: List[str] = []
        if escalation.level == "urgent":
            steps.append("Contact a veterinarian today (call ahead and describe what you see).")
        elif escalation.level == "contact_vet_soon":
            steps.append("Arrange a veterinary consultation within the next 24-48 hours and bring or share these photos.")
        elif escalation.level == "monitor":
            steps.append("Monitor over the next 24-48 hours: note appetite, energy, vomiting, and every stool.")
        else:
            steps.append("No action needed from this photo. Keep logging observations to build a baseline.")
        if cls.category in LOOSE_CATEGORIES or cls.category == "very hard / dry":
            steps.append("Make sure fresh water is always available and avoid new foods or treats until a veterinarian advises otherwise.")
        if comparison.n_previous < 3:
            steps.append("Log a few more observations (ideally across several days) so trends can be compared.")
        if quality_warnings:
            steps.append("For more reliable colour and shape, photograph the sample in bright, even daylight on a plain contrasting surface.")
        steps.append("Share this report with your veterinarian; it is a monitoring aid and cannot replace an examination or lab tests.")
        return steps


# --------------------------------------------------------------------------- #
class VeterinaryEscalationEngine:
    """Transparent, rule-based escalation. Every triggered rule is listed in the output."""

    def evaluate(self, cls: FecalClassification, findings: Sequence[Finding], comparison: HistoricalComparison,
                 symptoms: Sequence[str], refs: Dict[str, List[str]], llm_flags: Sequence[str] = ()) -> Escalation:
        """Return the escalation level, headline and the rules that produced it."""
        flags = flagged(findings)
        rules: List[Tuple[str, str, str]] = []  # (level, rule_id, reason)
        category = cls.category
        score = cls.score
        loose = category in LOOSE_CATEGORIES
        blood_like = "red_discoloration" in flags or "dark_or_black_stool" in flags or any(f in ("red_discoloration", "dark_or_black_stool") for f in llm_flags)

        if blood_like:
            rules.append(("contact_vet_soon", "R1_blood_like_colour",
                          "Red or black discolouration is visible; blood in stool warrants veterinary advice."))
        if blood_like and loose:
            rules.append(("urgent", "R2_blood_like_plus_loose", "Possible blood together with loose or watery stool."))
        if blood_like and comparison.repeated_abnormal:
            rules.append(("urgent", "R3_blood_like_repeated", "Concerning colour together with repeated abnormal observations this week."))
        if symptoms and (blood_like or loose or "foreign_material" in flags):
            rules.append(("urgent", "R4_symptoms_plus_stool_change",
                          "Reported symptoms (" + ", ".join(symptoms) + ") together with an abnormal stool observation."))
        elif symptoms:
            rules.append(("contact_vet_soon", "R5_symptoms_reported", "Reported symptoms: " + ", ".join(symptoms) + "."))
        if "foreign_material" in flags:
            rules.append(("contact_vet_soon", "R6_possible_foreign_material", "Possible non-stool material is visible."))
        if "pale_or_gray_stool" in flags:
            rules.append(("contact_vet_soon", "R7_pale_gray", "Pale gray or whitish stool is visible."))
        if "yellow_orange_stool" in flags:
            rules.append(("contact_vet_soon", "R8_yellow_orange", "Yellow-orange stool is visible."))
        if "mucus_like_coating" in flags and loose:
            rules.append(("contact_vet_soon", "R9_mucus_plus_loose", "Possible mucus together with loose stool."))
        elif "mucus_like_coating" in flags:
            rules.append(("monitor", "R10_possible_mucus", "A possible mucus-like coating is visible."))
        if comparison.repeated_abnormal and is_abnormal_consistency(category):
            rules.append(("contact_vet_soon", "R11_persistent_abnormal",
                          "Consistency has been outside the firm/soft-formed range on repeated observations this week."))
        if loose:
            rules.append(("monitor", "R12_loose_single", "Loose or watery stool on this observation."))
        if category == "very hard / dry":
            rules.append(("monitor", "R13_very_hard", "Very hard, dry stool on this observation."))
        if comparison.trend == "looser" and (comparison.score_delta or 0) >= 2:
            rules.append(("monitor", "R14_trend_looser", "Stool is markedly looser than this pet's recent pattern."))
        if comparison.repeated_flags:
            rules.append(("contact_vet_soon", "R15_repeated_flag", "The same visual flag recurred: " + ", ".join(comparison.repeated_flags) + "."))

        level = "none"
        for lvl, _rid, _why in rules:
            if ESCALATION_RANK[lvl] > ESCALATION_RANK[level]:
                level = lvl
        headline = {
            "none": "Nothing concerning was visible in this photo (a photo cannot rule out problems).",
            "monitor": "Keep an eye on this: monitor for 24-48 hours.",
            "contact_vet_soon": "Veterinary consultation recommended within 24-48 hours.",
            "urgent": "Contact a veterinarian today.",
        }[level]
        ref_ids: List[str] = []
        for topic in ("red", "black", "loose", "mucus", "pale", "yellow", "foreign", "hard"):
            ref_ids.extend(refs.get(topic, []))
        ref_ids = sorted(set(ref_ids)) if level != "none" else []
        reasons = [why for _l, _r, why in rules] or ["No rule was triggered by the observed features."]
        return Escalation(level=level, headline=headline, reasons=reasons,
                          rules_triggered=[rid for _l, rid, _w in rules], reference_ids=ref_ids)

    @staticmethod
    def inconclusive(reason: str) -> Escalation:
        """Escalation record for photos that could not be assessed."""
        return Escalation(
            level="inconclusive",
            headline="This photo could not be assessed, so no conclusion is drawn.",
            reasons=[reason],
            rules_triggered=["R0_input_not_assessable"],
        )
