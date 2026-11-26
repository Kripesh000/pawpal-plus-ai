"""Plain-text rendering of an AnalysisReport (the eight required outputs, in order)."""

from __future__ import annotations

from typing import List

from .models import AnalysisReport


def render_text(r: AnalysisReport, show_trace: bool = False) -> str:
    """Return a human-readable report."""
    out: List[str] = []
    add = out.append
    add("=" * 74)
    add(f"PawPal+ AI stool observation  |  {r.pet_name} ({r.species})  |  {r.created_at}")
    add(f"Photo: {r.image_name}  (source: {r.source_type})   Status: {r.status.upper()}")
    add("=" * 74)

    q = r.image_quality
    add("\n1. IMAGE QUALITY")
    if q is None:
        add(f"   Rejected before quality checks: {r.input_error}")
    else:
        add(f"   Usable: {'yes' if q.usable else 'NO'}   Quality score: {q.score:.2f}   "
            f"Stool-coloured subject found: {'yes' if q.subject_visible else 'no'}")
        m = q.metrics
        add(f"   Brightness {m['brightness']}, sharpness {m['sharpness']}, subject area {m['subject_fraction'] * 100:.1f}% of frame")
        if q.issues:
            add("   Blocking issues: " + ", ".join(q.issues))
        if q.warnings:
            add("   Warnings: " + ", ".join(q.warnings))
        add(f"   -> {q.recommendation}")

    if r.status != "completed":
        add("\nNo stool observations were generated because the photo could not be assessed.")
        _tail(out, r, show_trace)
        return "\n".join(out)

    c = r.classification
    add("\n2. CLASSIFICATION & SCORING  (heuristic estimate)")
    add(f"   Scale: {c.scale}")
    if c.score is not None:
        add(f"   Estimated score: {c.score}  -  {c.description}")
    add(f"   Category: {c.category}   Confidence: {c.confidence:.2f}")
    if c.alternatives:
        alts = ", ".join(
            (f"score {a['score']} ({a['probability']:.2f})" if "score" in a else f"{a['category']} ({a['probability']:.2f})")
            for a in c.alternatives)
        add(f"   Other plausible: {alts}")

    add("\n3. VISUAL OBSERVATIONS  (what is visible, not what it means)")
    for f in r.observations:
        mark = {"observed": " ", "possible": "!", "not_detected": "-", "uncertain": "?"}[f.status]
        add(f"   [{mark}] {f.name.replace('_', ' '):<22} ({f.status}, conf {f.confidence:.2f})  {f.detail}")

    add("\n4. CONFIDENCE")
    cf = r.confidence
    add(f"   Overall: {cf.overall:.2f} ({cf.level})   components: " + ", ".join(f"{k}={v}" for k, v in cf.components.items()))
    for cav in cf.caveats:
        add(f"   - {cav}")

    add("\n5. COMPARISON WITH PREVIOUS OBSERVATIONS")
    add(f"   Previous records: {r.comparison.n_previous}   Trend: {r.comparison.trend}")
    add(f"   {r.comparison.summary}")

    add("\n6. POSSIBLE HEALTH / DIETARY ASSOCIATIONS  (possible only; not diagnoses)")
    if r.associations:
        for a in r.associations:
            add(f"   * [{a.basis}] {a.statement}" + (f"  [refs: {', '.join(a.reference_ids)}]" if a.reference_ids else ""))
    else:
        add("   (none: nothing notable was observed)")

    add("\n7. RECOMMENDED NEXT STEPS")
    for i, s in enumerate(r.next_steps, 1):
        add(f"   {i}. {s}")

    _tail(out, r, show_trace)
    return "\n".join(out)


def _tail(out: List[str], r: AnalysisReport, show_trace: bool) -> None:
    e = r.escalation
    out.append("\n8. VETERINARY ESCALATION")
    out.append(f"   Level: {e.level.upper()}  -  {e.headline}")
    for reason in e.reasons:
        out.append(f"   - {reason}")
    if e.rules_triggered:
        out.append("   Rules: " + ", ".join(e.rules_triggered))
    if r.status == "completed":
        from .insights import STANDING_ADVICE

        out.append("   " + STANDING_ADVICE)
    if r.references:
        out.append("\nREFERENCES RETRIEVED")
        for ref in r.references:
            out.append(f"   [{ref.ref_id}] {ref.title} - {ref.source}\n        {ref.url if ref.url != 'none' else '(project note; no external URL)'}")
    if r.second_opinion:
        out.append(f"\nSECOND OPINION (multimodal model {r.second_opinion['model']}): agreement {r.second_opinion['agreement']}")
    out.append("\nSAFETY VALIDATION: " + ("passed" if r.validation and r.validation.passed else "FAILED")
               + (f" ({len(r.validation.redactions)} redaction(s))" if r.validation and r.validation.redactions else ""))
    if r.stored:
        out.append("Stored in the pet's stool history.")
    out.append("")
    for d in r.disclaimers:
        out.append(f"* {d}")
    if show_trace:
        out.append("\nEXECUTION TRACE")
        for s in r.trace:
            out.append(f"   {s.stage:<24} {s.status:<8} {s.duration_ms:>7.1f} ms  {s.detail}")
