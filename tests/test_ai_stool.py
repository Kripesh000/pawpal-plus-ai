"""Tests for the PawPal+ AI stool-monitoring package (uses SYNTHETIC images only)."""

import io
import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from PIL import Image

from pawpal_ai import ImageInput, StoolAnalysisPipeline
from pawpal_ai.cli import main as cli_main
from pawpal_ai.demo import build_demo_owner, run_demo, write_execution_log
from pawpal_ai.guardrails import AIResponseValidator, ReliabilityEvaluator, safety_violations
from pawpal_ai.history import PetHistoryManager
from pawpal_ai.imaging import ImageInputError, ImageProcessor, ImageQualityValidator
from pawpal_ai.integration import schedule_vet_followup
from pawpal_ai.knowledge import KnowledgeBase
from pawpal_ai.llm_backend import LLMUnavailable, LLMVisionBackend
from pawpal_ai.models import AnalysisReport, Escalation, Finding
from pawpal_ai.samples import render_sample, to_bytes
from pawpal_system import Owner, Pet

NOW = datetime(2026, 11, 24, 18, 0, 0)


def photo(kind, seed=900, **kw):
    return to_bytes(render_sample(kind, seed, **kw))


def run(data, species="Dog", pet=None, **kw):
    return StoolAnalysisPipeline().analyze(pet or Pet("T", species), ImageInput(data, "t.png"), persist=kw.pop("persist", False), now=NOW, **kw)


def flags(report):
    return {f.name for f in report.observations if f.status == "possible"}


# ---------------------------------------------------------------- input handling
@pytest.mark.parametrize("data,code", [
    (b"", "empty"),
    (b"not an image at all", "unreadable"),
    (to_bytes(Image.new("RGB", (40, 40))), "too_small"),
])
def test_processor_rejects_bad_input(data, code):
    with pytest.raises(ImageInputError) as exc:
        ImageProcessor().load(ImageInput(data))
    assert exc.value.code == code


def test_processor_rejects_unsupported_format_and_truncated():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200)).save(buf, format="GIF")
    with pytest.raises(ImageInputError) as exc:
        ImageProcessor().load(ImageInput(buf.getvalue()))
    assert exc.value.code == "unsupported_format"
    good = photo("log")
    with pytest.raises(ImageInputError):
        ImageProcessor().load(ImageInput(good[: len(good) // 3]))


def test_processor_handles_rgba_and_large_images_and_jpeg():
    rgba = Image.new("RGBA", (300, 300), (200, 190, 180, 128))
    loaded = ImageProcessor().load(ImageInput(to_bytes(rgba)))
    assert loaded.rgb.shape[2] == 3
    big = Image.new("RGB", (1600, 1200), (200, 190, 180))
    loaded = ImageProcessor().load(ImageInput(to_bytes(big, "JPEG")))
    assert max(loaded.rgb.shape[:2]) == 512 and loaded.fmt == "JPEG"


def test_image_source_type_validated():
    with pytest.raises(ValueError):
        ImageInput(b"x", source_type="drone")
    assert ImageInput(b"x", source_type="litter_box").source_type == "litter_box"


# ---------------------------------------------------------------- quality gate
@pytest.mark.parametrize("make,issue", [
    (lambda: photo("log", extra="blur"), "out_of_focus"),
    (lambda: photo("log", extra="dark"), "too_dark"),
    (lambda: photo("log", extra="overexposed"), "overexposed"),
    (lambda: photo("blank", 1), "flat_or_blank"),
    (lambda: photo("empty"), "no_subject_found"),
    (lambda: photo("blue_object"), "subject_not_stool_coloured"),
])
def test_quality_validator_flags_unusable_photos(make, issue):
    q, _ = ImageQualityValidator().assess(ImageProcessor().load(ImageInput(make())))
    assert not q.usable and issue in q.issues and "retake" in q.recommendation.lower()


def test_quality_validator_accepts_good_photo_on_each_background():
    for bg in ("tile", "grass", "concrete"):
        q, subject = ImageQualityValidator().assess(ImageProcessor().load(ImageInput(photo("log", background=bg))))
        assert q.usable and q.subject_visible and 0.02 < subject.fraction < 0.85


# ---------------------------------------------------------------- scoring
@pytest.mark.parametrize("kind,allowed", [("pellets", {1}), ("firm_log", {2, 3}), ("log", {2, 3, 4}),
                                          ("soft_log", {3, 4, 5}), ("piles", {4, 5, 6}), ("spots", {5, 6}), ("puddle", {7})])
def test_dog_scores_are_in_expected_band_for_each_synthetic_class(kind, allowed):
    r = run(photo(kind, seed=910))
    assert r.status == "completed" and r.classification.score in allowed


def test_cat_gets_category_but_never_a_numeric_score():
    r = run(photo("log"), species="Cat")
    assert r.classification.score is None and r.classification.category and "No numeric scale" in r.classification.scale
    assert all("score" not in a for a in r.classification.alternatives)


# ---------------------------------------------------------------- observations
def test_observation_flags():
    assert flags(run(photo("firm_log", extra="red_streaks"))) == {"red_discoloration"}
    assert "dark_or_black_stool" in flags(run(photo("log", color="black")))
    assert "pale_or_gray_stool" in flags(run(photo("log", color="gray", background="concrete")))
    assert "mucus_like_coating" in flags(run(photo("log", extra="mucus", background="concrete")))
    assert "foreign_material" in flags(run(photo("firm_log", extra="foreign")))
    assert flags(run(photo("firm_log"))) == set()


def test_not_detected_findings_never_claim_absence():
    r = run(photo("firm_log"))
    nd = [f for f in r.observations if f.status == "not_detected"]
    assert nd and all("rule out" in f.detail or "not" in f.detail.lower() or "No " in f.detail for f in nd)
    assert all(f.confidence <= 0.6 for f in nd)


# ---------------------------------------------------------------- pipeline: rejection paths
def test_rejected_input_has_no_observations_and_is_not_stored():
    pet = Pet("T", "Dog")
    r = StoolAnalysisPipeline().analyze(pet, ImageInput(b"junk"), now=NOW)
    assert r.status == "rejected_input" and r.observations == [] and r.classification is None
    assert r.escalation.level == "inconclusive" and not r.stored and pet.stool_records == []
    assert r.validation.passed and r.disclaimers


def test_rejected_quality_has_no_fabricated_observations():
    pet = Pet("T", "Dog")
    r = StoolAnalysisPipeline().analyze(pet, ImageInput(photo("log", extra="blur")), now=NOW)
    assert r.status == "rejected_quality" and r.observations == [] and r.classification is None
    assert r.image_quality and not r.image_quality.usable and r.next_steps and not r.stored


def test_unsupported_species_and_unknown_symptom():
    r = StoolAnalysisPipeline().analyze(Pet("H", "Hamster"), ImageInput(photo("log")), now=NOW)
    assert r.status == "rejected_input" and "dogs and cats" in r.input_error
    r = run(photo("log"), symptoms=("itchy",))
    assert r.status == "rejected_input"


# ---------------------------------------------------------------- escalation rules
def test_escalation_levels():
    assert run(photo("firm_log")).escalation.level == "none"
    assert run(photo("pellets")).escalation.level == "monitor"
    assert run(photo("firm_log", extra="red_streaks")).escalation.level == "contact_vet_soon"
    r = run(photo("puddle", extra="red_streaks"))
    assert r.escalation.level == "urgent" and "R2_blood_like_plus_loose" in r.escalation.rules_triggered
    assert run(photo("firm_log"), symptoms=("vomiting",)).escalation.level == "contact_vet_soon"
    assert run(photo("piles"), symptoms=("lethargy",)).escalation.level == "urgent"


def test_every_report_has_eight_outputs_and_disclaimers():
    r = run(photo("piles"))
    assert (r.image_quality and r.classification and r.observations and r.confidence and r.comparison is not None
            and r.next_steps and r.escalation and r.associations is not None)
    assert len(r.disclaimers) == 3 and r.validation.passed
    json.dumps(r.to_dict(), default=str)  # structured and serialisable


# ---------------------------------------------------------------- history
def seeded(specs):
    pet = Pet("T", "Dog")
    for i, (days, score, cat) in enumerate(specs):
        PetHistoryManager.add(pet, {"report_id": f"s{i}", "timestamp": (NOW - timedelta(days=days)).isoformat(), "image_sha256": "",
                                    "species": "dog", "score": score, "category": cat, "color": "brown", "flags": [],
                                    "confidence": 0.6, "escalation": "none"})
    return pet


def test_history_no_history_stable_and_looser():
    assert run(photo("log")).comparison.trend == "no_history"
    firm = [(6, 2, "firm, well formed"), (4, 2, "firm, well formed"), (2, 2, "firm, well formed")]
    assert run(photo("firm_log"), pet=seeded(firm)).comparison.trend == "stable"
    r = run(photo("puddle"), pet=seeded(firm))
    assert r.comparison.trend == "looser" and r.comparison.score_delta >= 2


def test_repeated_loose_triggers_persistent_rule():
    pet = seeded([(5, 6, "loose / unformed"), (3, 5, "loose / unformed"), (1, 6, "loose / unformed")])
    r = run(photo("piles"), pet=pet)
    assert r.comparison.repeated_abnormal and "R11_persistent_abnormal" in r.escalation.rules_triggered


def test_old_records_outside_window_do_not_count_as_repeated():
    pet = seeded([(30, 6, "loose / unformed"), (25, 6, "loose / unformed")])
    assert not run(photo("piles"), pet=pet).comparison.repeated_abnormal


def test_duplicate_photo_not_double_counted_and_history_persists(tmp_path):
    owner = Owner("O")
    pet = owner.add_pet(Pet("Luna", "Dog"))
    pipe = StoolAnalysisPipeline()
    data = photo("log")
    first = pipe.analyze(pet, ImageInput(data), now=NOW)
    second = pipe.analyze(pet, ImageInput(data), now=NOW + timedelta(hours=1))
    assert first.stored and not second.stored and second.comparison.trend == "duplicate"
    path = str(tmp_path / "d.json")
    owner.save_to_json(path)
    assert len(Owner.load_from_json(path).get_pet("Luna").stool_records) == 1


def test_legacy_json_without_stool_records_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "owner": {"name": "A", "pets": [{"name": "P", "species": "Dog", "tasks": []}]}}))
    assert Owner.load_from_json(str(path)).pets[0].stool_records == []


def test_history_record_validation():
    with pytest.raises(ValueError):
        PetHistoryManager.add(Pet("P", "Dog"), {"report_id": "x"})


# ---------------------------------------------------------------- diet / medication context
def test_diet_and_medication_context_is_hedged_and_appears_only_when_notable():
    owner = build_demo_owner()
    luna = owner.get_pet("Luna")
    r = StoolAnalysisPipeline().analyze(luna, ImageInput(photo("piles")), persist=False, now=NOW)
    ctx = [a for a in r.associations if a.basis == "history_context"]
    text = " ".join(a.statement for a in ctx)
    assert "salmon" in text and "Heartworm" in text and "Do not change or stop any medication" in text
    assert "caused" not in text
    normal = StoolAnalysisPipeline().analyze(luna, ImageInput(photo("firm_log")), persist=False, now=NOW)
    assert not [a for a in normal.associations if a.basis == "history_context"]


# ---------------------------------------------------------------- safety filter / validator
@pytest.mark.parametrize("text", [
    "Your dog has parvovirus.", "This is a diagnosis of colitis.", "Give 5 mg twice a day.",
    "Stop the medication immediately.", "It is definitely caused by an ulcer.", "Colitis is present.",
])
def test_unsafe_text_blocked(text):
    assert safety_violations(text)


@pytest.mark.parametrize("text", [
    "Loose stool can be associated with a recent food change.", "Only a veterinarian can establish a diagnosis.",
    "Do not change or stop any medication without veterinary advice.", "Contact a veterinarian within 24-48 hours.",
])
def test_safe_text_allowed(text):
    assert not safety_violations(text)


def test_enforce_redacts_unsafe_items_and_adds_disclaimers():
    r = run(photo("piles"))
    from pawpal_ai.models import Association
    r.associations.append(Association("Your dog has giardia.", "visual_only"))
    r.next_steps.append("Give 10 mg of antibiotic.")
    res = AIResponseValidator().enforce(r)
    assert res.passed and len(res.redactions) == 2
    assert not any("giardia" in a.statement for a in r.associations) and not any("10 mg" in s for s in r.next_steps)


def test_structural_validation_catches_bad_reports():
    r = run(photo("log"))
    v = AIResponseValidator()
    r.classification.score = 11
    assert any("1-7" in i for i in v.structural_issues(r))
    r.classification.score = None
    assert any("1-7" in i for i in v.structural_issues(r))  # dog without a score
    r.classification.score, r.classification.confidence = 3, 1.4
    assert any("confidence" in i for i in v.structural_issues(r))
    r.classification.confidence = 0.5
    r.escalation = Escalation("panic", "x")
    assert any("invalid escalation" in i for i in v.structural_issues(r))
    cat = run(photo("log"), species="Cat")
    cat.classification.score = 3
    assert any("cats" in i for i in v.structural_issues(cat))
    rej = run(b"junk")
    rej.observations.append(Finding("color", "observed", 0.5, "brown"))
    assert any("rejected analysis" in i for i in v.structural_issues(rej))


def test_overconfident_finding_is_flagged():
    r = run(photo("log"))
    r.observations[0].confidence = 0.99
    assert any("implausibly high" in i for i in AIResponseValidator().structural_issues(r))


# ---------------------------------------------------------------- reliability
def test_stability_and_confidence_summary():
    loaded = ImageProcessor().load(ImageInput(photo("soft_log", seed=930)))
    pipe = StoolAnalysisPipeline()
    q, subject = pipe.quality.assess(loaded)
    feats, _ = pipe.analyzer.analyze(subject, q)
    cls = pipe.scorer.classify(feats, "dog", q)
    st = ReliabilityEvaluator().stability(loaded, pipe.quality, pipe.analyzer, pipe.scorer, "dog", cls.score, cls.category)
    assert 0.0 <= st["agreement"] <= 1.0 and len(st["variants"]) == 4
    summary = ReliabilityEvaluator().summarise(q, cls, q.subject_confidence, st, None, False)
    assert 0 <= summary.overall <= 0.9 and summary.level in ("low", "moderate", "high")


def test_confidence_is_capped_and_low_confidence_gets_caveat():
    r = run(photo("log"))
    assert r.confidence.overall <= 0.9
    assert any("heuristic" in c for c in r.confidence.caveats)


# ---------------------------------------------------------------- knowledge base
def test_knowledge_retrieval_species_filter_and_missing_dir():
    kb = KnowledgeBase()
    assert len(kb) > 5
    assert kb.retrieve("fecal score consistency scale", "dog", top_k=1)[0].ref_id == "purina-fecal-score-dog"  # regression
    assert not any(h.ref_id == "royalcanin-fecal-score-cat" for h in kb.retrieve("fecal score scale", "dog", top_k=5))
    cat_blood = kb.retrieve("bright red blood streaks", "cat", top_k=1)
    assert cat_blood and "dog-specific" in cat_blood[0].source
    missing = KnowledgeBase(directory="/no/such/dir")
    assert len(missing) == 0 and missing.load_error and missing.retrieve("blood", "dog") == []
    assert KnowledgeBase.empty().retrieve("blood") == []


def test_citations_present_in_report():
    r = run(photo("firm_log", extra="red_streaks"))
    assert any(ref.ref_id == "akc-blood-in-stool" for ref in r.references)
    assert all(ref.url for ref in r.references)


# ---------------------------------------------------------------- optional multimodal backend (FAKE client)
class FakeClient:
    def __init__(self, text):
        self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="text", text=text)]))


def llm_pipeline(text):
    return StoolAnalysisPipeline(llm=LLMVisionBackend(client=FakeClient(text), model="fake"))


GOOD = {"image_clear": True, "consistency_score": 3, "color": "brown", "shape": "log", "flags": [], "confidence": 0.6,
        "uncertainty_notes": "Estimate from one photo."}


def test_second_opinion_agreement_and_disagreement():
    r = llm_pipeline("Here you go: " + json.dumps(GOOD)).analyze(Pet("T", "Dog"), ImageInput(photo("log")), persist=False, now=NOW)
    assert r.second_opinion and r.second_opinion["agreement"] >= 0.5
    bad = dict(GOOD, image_clear=False, consistency_score=None)
    r = llm_pipeline(json.dumps(bad)).analyze(Pet("T", "Dog"), ImageInput(photo("log")), persist=False, now=NOW)
    assert r.second_opinion["agreement"] == 0.0 and any("second opinion" in c for c in r.confidence.caveats)


def test_second_opinion_extra_blood_flag_raises_escalation():
    payload = dict(GOOD, flags=["red_discoloration"])
    r = llm_pipeline(json.dumps(payload)).analyze(Pet("T", "Dog"), ImageInput(photo("firm_log")), persist=False, now=NOW)
    assert r.escalation.level == "contact_vet_soon" and "R1_blood_like_colour" in r.escalation.rules_triggered


@pytest.mark.parametrize("text", [json.dumps(dict(GOOD, uncertainty_notes="Your dog has parvovirus.")), "no json here", "{not valid json}",
                                  json.dumps({"image_clear": True})])
def test_invalid_or_unsafe_model_output_is_ignored(text):
    r = llm_pipeline(text).analyze(Pet("T", "Dog"), ImageInput(photo("firm_log")), persist=False, now=NOW)
    assert r.status == "completed" and r.second_opinion is None and r.escalation.level == "none"
    assert [t.status for t in r.trace if t.stage == "second_opinion"][0] in ("warn", "skipped")


def test_llm_unconfigured_is_skipped(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not LLMVisionBackend.is_configured()
    with pytest.raises(LLMUnavailable):
        LLMVisionBackend().describe(ImageProcessor().load(ImageInput(photo("log"))), "dog", [], "")


# ---------------------------------------------------------------- integration with the scheduler + CLI
def test_schedule_vet_followup_is_opt_in_and_idempotent():
    owner = build_demo_owner()
    luna = owner.get_pet("Luna")
    pipe = StoolAnalysisPipeline()
    normal = pipe.analyze(luna, ImageInput(photo("firm_log")), persist=False, now=NOW)
    assert schedule_vet_followup(owner, luna, normal) is None
    before = len(luna.tasks)
    soon = pipe.analyze(luna, ImageInput(photo("firm_log", extra="red_streaks")), persist=False, now=NOW)
    assert len(luna.tasks) == before  # analysis alone never edits the schedule
    task = schedule_vet_followup(owner, luna, soon)
    assert task and task.category == "Vet" and task.priority == "High" and task.date == "2026-11-25"
    assert schedule_vet_followup(owner, luna, soon) is None  # no duplicate
    urgent = pipe.analyze(luna, ImageInput(photo("puddle", extra="red_streaks")), persist=False, now=NOW)
    assert schedule_vet_followup(owner, luna, urgent).date == "2026-11-24"


def test_vet_task_avoids_busy_slots():
    owner = build_demo_owner()
    luna = owner.get_pet("Luna")
    from pawpal_system import Task
    luna.add_task(Task("Busy", "08:00", "2026-11-25", duration_mins=60))
    soon = StoolAnalysisPipeline().analyze(luna, ImageInput(photo("firm_log", extra="red_streaks")), persist=False, now=NOW)
    assert schedule_vet_followup(owner, luna, soon).time == "09:00"


def test_demo_runs_all_scenarios_and_writes_log(tmp_path):
    runs = run_demo(verbose=False)
    statuses = [r.status for _l, r in runs]
    assert statuses.count("completed") == 5 and "rejected_quality" in statuses and "rejected_input" in statuses
    assert all(r.validation.passed for _l, r in runs)
    log = tmp_path / "ai.md"
    log.write_text("# existing content\n")
    write_execution_log(runs, str(log))
    write_execution_log(runs, str(log))  # idempotent replace
    text = log.read_text()
    assert text.startswith("# existing content") and text.count("BEGIN RUNTIME TRACE") == 1 and "| input_validation |" in text


def test_cli_analyze_json_and_errors(tmp_path, capsys):
    img = tmp_path / "p.png"
    img.write_bytes(photo("firm_log", extra="red_streaks"))
    data = tmp_path / "data.json"
    assert cli_main(["analyze", "--image", str(img), "--pet", "Luna", "--data", str(data), "--json", "--schedule-vet"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[: out.rindex("}") + 1])
    assert payload["escalation"]["level"] == "contact_vet_soon"
    saved = Owner.load_from_json(str(data))
    assert len(saved.get_pet("Luna").stool_records) == 1 and any(t.category == "Vet" for t in saved.get_pet("Luna").tasks)
    assert cli_main(["analyze", "--image", str(img), "--pet", "Nobody"]) == 2
    assert cli_main(["analyze", "--image", str(tmp_path / "missing.png"), "--pet", "Luna"]) == 2
    assert cli_main(["history", "--pet", "Luna", "--data", str(data)]) == 0


# ---------------------------------------------------------------- Streamlit tab (STUB: real Streamlit not installed here)
class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeStreamlit:
    """Records calls; stands in for the real streamlit package in this sandbox."""

    def __init__(self, upload=None, press=("stool_go",)):
        self.session_state, self.calls, self.upload, self.press = {}, [], upload, set(press)

    def __getattr__(self, name):
        def fn(*a, **k):
            self.calls.append((name, a, k))
            if name == "selectbox": return a[1][0]
            if name == "file_uploader": return self.upload
            if name == "multiselect": return []
            if name == "checkbox": return k.get("value", False)
            if name == "button": return k.get("key") in self.press and not k.get("disabled", False)
            if name == "expander": return _Ctx()
            return None
        return fn


def test_streamlit_tab_renders_with_stub():
    owner = build_demo_owner()
    from pawpal_ai.ui import render_stool_tab
    up = SimpleNamespace(getvalue=lambda: photo("firm_log", extra="red_streaks"), name="red.png")
    st = FakeStreamlit(upload=up, press=("stool_go", "stool_book"))
    render_stool_tab(st, owner)
    names = [c[0] for c in st.calls]
    assert "table" in names and "dataframe" in names and st.session_state["stool_report"].escalation.level == "contact_vet_soon"
    assert any(t.category == "Vet" for t in owner.get_pet("Luna").tasks)
    empty = FakeStreamlit()
    render_stool_tab(empty, Owner("Nobody"))
    assert any(c[0] == "info" for c in empty.calls)
    bad = FakeStreamlit(upload=SimpleNamespace(getvalue=lambda: b"junk", name="x.png"))
    render_stool_tab(bad, build_demo_owner())
    assert bad.session_state["stool_report"].status == "rejected_input"


# ---------------------------------------------------------------- documentation stays honest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_embeds_the_same_mermaid_diagram_as_architecture_mmd():
    source = open(os.path.join(ROOT, "architecture.mmd"), encoding="utf-8").read().strip()
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "```mermaid\n" + source in readme


def test_required_deliverables_exist_and_diagram_names_real_components():
    for name in ("README.md", "architecture.mmd", "model_card.md", "ai_interactions.md", "ai_collaboration_reflection.md",
                 "evaluate.py", ".env.example"):
        assert os.path.exists(os.path.join(ROOT, name)), name
    source = open(os.path.join(ROOT, "architecture.mmd"), encoding="utf-8").read()
    import pawpal_ai.guardrails as g, pawpal_ai.imaging as i, pawpal_ai.insights as n, pawpal_ai.history as h
    for cls in ("ImageProcessor", "ImageQualityValidator", "StoolAnalyzer", "FecalScoringEngine", "ReliabilityEvaluator",
                "PetHistoryManager", "DietCorrelationEngine", "HealthInsightGenerator", "VeterinaryEscalationEngine",
                "AIResponseValidator", "KnowledgeBase", "LLMVisionBackend"):
        assert cls in source
    for mod, cls in ((i, "ImageProcessor"), (g, "AIResponseValidator"), (n, "VeterinaryEscalationEngine"), (h, "PetHistoryManager")):
        assert hasattr(mod, cls)


def test_invalid_capture_timestamp_is_noted_not_crashed():
    r = StoolAnalysisPipeline().analyze(Pet("T", "Dog"), ImageInput(photo("log"), captured_at="yesterday-ish"), persist=False, now=NOW)
    step = [s for s in r.trace if s.stage == "input_validation"][0]
    assert r.status == "completed" and step.status == "warn" and "captured_at" in step.detail
