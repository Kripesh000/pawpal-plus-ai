"""Streamlit tab for stool analysis. Takes the `st` module as an argument so it can be exercised with a stub."""

from __future__ import annotations

from typing import Any, Callable, Optional

from pawpal_system import Owner

from .history import PetHistoryManager
from .insights import SYMPTOMS
from .integration import schedule_vet_followup
from .models import ImageInput
from .pipeline import StoolAnalysisPipeline

LEVEL_ICON = {"inconclusive": "❔", "none": "🟢", "monitor": "🟡", "contact_vet_soon": "🟠", "urgent": "🔴"}


def render_stool_tab(st: Any, owner: Owner, on_change: Optional[Callable[[str, str], None]] = None) -> None:
    """Draw the stool-analysis tab: pet, upload, analyse, eight output sections, history, scheduling."""
    st.subheader("🔬 Stool photo analysis")
    st.warning("Monitoring aid only. This tool cannot diagnose disease; parasites, bacteria and microscopic blood cannot be seen "
               "in photos. Only a veterinarian can diagnose. Results come from a heuristic prototype, not a trained model.")
    if not owner.pets:
        st.info("Add a pet in the Pets tab first.")
        return
    pet_name = st.selectbox("Pet", [p.name for p in owner.pets], key="stool_pet")
    pet = owner.get_pet(pet_name)
    upload = st.file_uploader("Photo of the stool sample (PNG / JPG / WEBP / BMP)", type=["png", "jpg", "jpeg", "webp", "bmp"], key="stool_upload")
    symptoms = st.multiselect("Symptoms right now (optional)", list(SYMPTOMS), key="stool_symptoms")
    store = st.checkbox("Store this result in the pet's history", value=True, key="stool_store")

    if st.button("Analyze photo", disabled=upload is None, key="stool_go"):
        report = StoolAnalysisPipeline().analyze(
            pet, ImageInput(upload.getvalue(), upload.name, "file_upload"), symptoms=list(symptoms), persist=bool(store))
        st.session_state["stool_report"] = report
        if report.stored and on_change:
            on_change("success", f"Stored in {pet.name}'s stool history.")

    report = st.session_state.get("stool_report")
    if report is not None and report.pet_id == pet.id:
        _render_report(st, owner, pet, report)

    st.markdown("**History**")
    records = PetHistoryManager.records(pet)
    if records:
        st.dataframe([{"When": r["timestamp"][:16], "Score": r.get("score"), "Category": r.get("category"),
                       "Colour": r.get("color"), "Flags": ", ".join(r.get("flags", [])) or "-",
                       "Escalation": r.get("escalation")} for r in records], hide_index=True)
    else:
        st.caption("No stool records yet.")


def _render_report(st: Any, owner: Owner, pet: Any, r: Any) -> None:
    e = r.escalation
    st.markdown(f"### {LEVEL_ICON.get(e.level, '')} {e.headline}")
    q = r.image_quality
    st.markdown("**1. Image quality**")
    if q is None:
        st.error(r.input_error)
    else:
        (st.success if q.usable else st.error)(f"Usable: {q.usable} · score {q.score:.2f} · {q.recommendation}")
    if r.status != "completed":
        st.info("No observations were generated because the photo could not be assessed.")
        return
    c = r.classification
    st.markdown("**2. Classification**")
    st.write(f"{c.scale}")
    st.write((f"Estimated score **{c.score}**: " if c.score is not None else "") + f"{c.description} · category **{c.category}** · confidence {c.confidence:.2f}")
    st.markdown("**3. Visual observations**")
    st.table([{"Observation": f.name.replace("_", " "), "Status": f.status, "Confidence": f.confidence, "Detail": f.detail} for f in r.observations])
    st.markdown(f"**4. Confidence:** {r.confidence.overall:.2f} ({r.confidence.level})")
    for cav in r.confidence.caveats:
        st.caption(f"• {cav}")
    st.markdown(f"**5. Comparison with previous observations:** {r.comparison.summary}")
    st.markdown("**6. Possible associations (not diagnoses)**")
    for a in r.associations:
        st.write(f"• {a.statement}")
    if not r.associations:
        st.caption("None: nothing notable was observed.")
    st.markdown("**7. Recommended next steps**")
    for s in r.next_steps:
        st.write(f"• {s}")
    st.markdown("**8. Veterinary escalation**")
    for reason in e.reasons:
        st.write(f"• {reason}")
    with st.expander("References retrieved"):
        for ref in r.references:
            st.write(f"[{ref.title}]({ref.url}) — {ref.source}" if ref.url.startswith("http") else f"{ref.title} — {ref.source}")
    with st.expander("Execution trace"):
        st.table([{"Stage": s.stage, "Status": s.status, "ms": s.duration_ms, "Record": s.detail} for s in r.trace])
    if e.level in ("contact_vet_soon", "urgent") and st.button("Add a vet consult to the schedule", key="stool_book"):
        task = schedule_vet_followup(owner, pet, r)
        st.success(f"Added: {task.description} on {task.date} at {task.time}") if task else st.info("No task added (already scheduled or no free slot).")
    for d in r.disclaimers:
        st.caption(d)
