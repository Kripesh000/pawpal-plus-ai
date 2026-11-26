"""StoolAnalysisPipeline: the staged workflow that ties every component together.

Stages (each leaves an auditable StepRecord in report.trace; no private reasoning is stored):
  input_validation -> quality_validation -> observation_extraction -> fecal_scoring ->
  stability_check -> reference_retrieval -> second_opinion(optional) -> historical_comparison ->
  diet_medication_context -> insight_generation -> escalation -> safety_validation -> persist
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

from pawpal_system import Pet

from .analysis import StoolAnalyzer
from .guardrails import AIResponseValidator, ReliabilityEvaluator
from .history import PetHistoryManager
from .imaging import ImageInputError, ImageProcessor, ImageQualityValidator
from .insights import (SYMPTOMS, DietCorrelationEngine, HealthInsightGenerator, VeterinaryEscalationEngine,
                       flagged)
from .knowledge import KnowledgeBase
from .llm_backend import LLMUnavailable, LLMVisionBackend
from .models import SUPPORTED_SPECIES, AnalysisReport, ImageInput, RetrievedReference, StepRecord
from .scoring import FecalScoringEngine

TOPIC_QUERIES: Dict[str, str] = {
    "scale": "fecal score consistency scale",
    "limits": "photo limitations parasites microscopic blood diagnosis",
    "hard": "very hard dry pellets constipation",
    "loose": "diarrhea loose watery duration red flags vomiting",
    "red": "bright red blood streaks contact veterinarian",
    "black": "black tarry stool melena upper digestive tract",
    "pale": "white grey pale stool digestion",
    "yellow": "yellow orange stool liver",
    "mucus": "mucus slime colon irritation",
    "foreign": "foreign bodies swallowed",
    "diet": "diet change food switch gradual",
}
FLAG_TOPIC = {"red_discoloration": "red", "dark_or_black_stool": "black", "pale_or_gray_stool": "pale",
              "yellow_orange_stool": "yellow", "mucus_like_coating": "mucus", "foreign_material": "foreign"}


class StoolAnalysisPipeline:
    """Runs the full analysis for one pet and one photo; components are injectable for testing."""

    def __init__(self, processor: Optional[ImageProcessor] = None, quality: Optional[ImageQualityValidator] = None,
                 analyzer: Optional[StoolAnalyzer] = None, scorer: Optional[FecalScoringEngine] = None,
                 knowledge: Optional[KnowledgeBase] = None, history: Optional[PetHistoryManager] = None,
                 diet: Optional[DietCorrelationEngine] = None, insights: Optional[HealthInsightGenerator] = None,
                 escalation: Optional[VeterinaryEscalationEngine] = None, validator: Optional[AIResponseValidator] = None,
                 reliability: Optional[ReliabilityEvaluator] = None, llm: Optional[LLMVisionBackend] = None,
                 clock: Callable[[], datetime] = datetime.now) -> None:
        self.processor = processor or ImageProcessor()
        self.quality = quality or ImageQualityValidator()
        self.analyzer = analyzer or StoolAnalyzer()
        self.scorer = scorer or FecalScoringEngine()
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()
        self.history = history or PetHistoryManager()
        self.diet = diet or DietCorrelationEngine()
        self.insights = insights or HealthInsightGenerator()
        self.escalation = escalation or VeterinaryEscalationEngine()
        self.validator = validator or AIResponseValidator()
        self.reliability = reliability or ReliabilityEvaluator()
        self.llm = llm  # optional; None = heuristic only
        self.clock = clock

    # ------------------------------------------------------------------ #
    def analyze(self, pet: Pet, image: ImageInput, species: Optional[str] = None, symptoms: Sequence[str] = (),
                persist: bool = True, now: Optional[datetime] = None) -> AnalysisReport:
        """Analyse one photo for `pet`; always returns a validated AnalysisReport (never raises on bad input)."""
        trace: List[StepRecord] = []
        species_name = (species or pet.species or "").strip().lower()
        bad_timestamp = False
        try:
            ts = datetime.fromisoformat(image.captured_at) if image.captured_at else (now or self.clock())
        except ValueError:
            ts, bad_timestamp = now or self.clock(), True
        report = AnalysisReport(
            report_id=uuid.uuid4().hex[:10], created_at=ts.isoformat(timespec="seconds"), pet_id=pet.id,
            pet_name=pet.name, species=species_name, image_name=image.name, source_type=image.source_type,
            status="completed", trace=trace, symptoms=list(symptoms),
        )

        def step(stage: str, status: str, detail: str, t0: float) -> None:
            trace.append(StepRecord(stage, status, detail, round((time.perf_counter() - t0) * 1000, 1)))

        def reject(status: str, reason: str) -> AnalysisReport:
            report.status = status
            report.input_error = reason if status == "rejected_input" else ""
            report.escalation = VeterinaryEscalationEngine.inconclusive(reason)
            report.next_steps = ["Please provide a new photo." if status == "rejected_input" else report.next_steps[0]]
            report.validation = self.validator.enforce(report)
            trace.append(StepRecord("report", "stopped", f"{status}: no observations were generated", 0.0))
            return report

        # 1. input validation ------------------------------------------------
        t0 = time.perf_counter()
        bad_symptoms = [s for s in symptoms if s not in SYMPTOMS]
        if species_name not in SUPPORTED_SPECIES:
            step("input_validation", "stopped", f"unsupported species {species_name!r}", t0)
            return reject("rejected_input", f"Only dogs and cats are supported (got {species_name or 'none'}).")
        if bad_symptoms:
            step("input_validation", "stopped", f"unknown symptoms {bad_symptoms}", t0)
            return reject("rejected_input", f"Unknown symptom(s) {bad_symptoms}; allowed: {list(SYMPTOMS)}.")
        try:
            loaded = self.processor.load(image)
        except ImageInputError as exc:
            step("input_validation", "stopped", f"{exc.code}: {exc}", t0)
            return reject("rejected_input", str(exc))
        report.image_sha256 = loaded.sha256
        step("input_validation", "warn" if bad_timestamp else "ok",
             f"{loaded.fmt} {loaded.original_size[0]}x{loaded.original_size[1]}, source={image.source_type}, species={species_name}"
             + ("; captured_at was not a valid ISO timestamp, current time used" if bad_timestamp else ""), t0)

        # 2. quality validation ---------------------------------------------
        t0 = time.perf_counter()
        quality, subject = self.quality.assess(loaded)
        report.image_quality = quality
        if not quality.usable:
            step("quality_validation", "stopped", f"issues={quality.issues}, score={quality.score}", t0)
            report.next_steps = [quality.recommendation]
            return reject("rejected_quality", "Photo quality too low: " + ", ".join(quality.issues) + ".")
        step("quality_validation", "warn" if quality.warnings else "ok",
             f"score={quality.score}, warnings={quality.warnings or 'none'}", t0)

        # 3. observation extraction ----------------------------------------
        t0 = time.perf_counter()
        feats, findings = self.analyzer.analyze(subject, quality)
        report.observations = findings
        flags = flagged(findings)
        step("observation_extraction", "ok",
             f"method={self.analyzer.backend.name}; flags={sorted(flags) or 'none'}; pieces={feats.n_pieces}", t0)

        # 4. fecal scoring --------------------------------------------------
        t0 = time.perf_counter()
        cls = self.scorer.classify(feats, species_name, quality)
        report.classification = cls
        step("fecal_scoring", "ok", f"score={cls.score}, category={cls.category}, confidence={cls.confidence}", t0)

        # 5. stability check ------------------------------------------------
        t0 = time.perf_counter()
        stability = self.reliability.stability(loaded, self.quality, self.analyzer, self.scorer, species_name,
                                               cls.score, cls.category)
        step("stability_check", "ok" if stability["agreement"] >= 0.75 else "warn",
             f"agreement={stability['agreement']} across {len(stability['variants'])} perturbations", t0)

        # 6. reference retrieval (RAG) ---------------------------------------
        t0 = time.perf_counter()
        topics = ["scale", "limits"]
        loose = cls.category in ("loose / unformed", "watery / liquid")
        if cls.category == "very hard / dry":
            topics.append("hard")
        if loose:
            topics += ["loose", "diet"]
        topics += [FLAG_TOPIC[n] for n in flags if n in FLAG_TOPIC]
        refs: Dict[str, List[str]] = {}
        gathered: Dict[tuple, RetrievedReference] = {}
        for topic in dict.fromkeys(topics):
            hits = self.knowledge.retrieve(TOPIC_QUERIES[topic], species_name, top_k=1)
            refs[topic] = [h.ref_id for h in hits]
            for h in hits:
                gathered[(h.ref_id, h.section)] = h
        report.references = list(gathered.values())
        if len(self.knowledge) == 0:
            step("reference_retrieval", "warn", self.knowledge.load_error or "knowledge base empty; guidance is uncited", t0)
        else:
            step("reference_retrieval", "ok", f"{len(report.references)} passages for topics {list(dict.fromkeys(topics))}", t0)

        # 7. optional multimodal second opinion ----------------------------
        llm_agreement: Optional[float] = None
        llm_flags: List[str] = []
        if self.llm is not None:
            t0 = time.perf_counter()
            try:
                payload = self.llm.describe(loaded, species_name, report.references, "see comparison")
                ok, problems = self.validator.validate_llm_payload(payload)
                if not ok:
                    step("second_opinion", "warn", f"response rejected by validator: {problems}", t0)
                else:
                    heur = {n for n in flags}
                    theirs = set(payload["flags"])
                    llm_flags = sorted(theirs - heur)
                    if not payload["image_clear"]:
                        llm_agreement = 0.0
                    else:
                        score_ok = True if species_name == "cat" or payload["consistency_score"] is None else abs(payload["consistency_score"] - (cls.score or 0)) <= 1
                        union = heur | theirs
                        jacc = 1.0 if not union else len(heur & theirs) / len(union)
                        llm_agreement = 0.5 * float(score_ok) + 0.5 * jacc
                    report.second_opinion = {"model": self.llm.model, "payload": payload, "agreement": llm_agreement}
                    step("second_opinion", "ok", f"agreement={llm_agreement:.2f}; extra flags={llm_flags or 'none'}", t0)
            except LLMUnavailable as exc:
                step("second_opinion", "skipped", f"unavailable: {exc}", t0)
        else:
            trace.append(StepRecord("second_opinion", "skipped", "no multimodal backend configured", 0.0))

        # 8. confidence ------------------------------------------------------
        report.confidence = self.reliability.summarise(
            quality, cls, quality.subject_confidence, stability, llm_agreement,
            "dim_lighting_colour_unreliable" in quality.warnings)
        return self._finish(report, pet, feats, flags, refs, symptoms, llm_flags, persist, ts, trace)

    # ------------------------------------------------------------------ #
    def _finish(self, report, pet, feats, flags, refs, symptoms, llm_flags, persist, ts, trace) -> AnalysisReport:
        cls = report.classification

        def step(stage: str, status: str, detail: str, t0: float) -> None:
            trace.append(StepRecord(stage, status, detail, round((time.perf_counter() - t0) * 1000, 1)))

        # 9. historical comparison ------------------------------------------
        t0 = time.perf_counter()
        current = report.to_history_record()
        report.comparison = self.history.compare(pet, current, ts)
        step("historical_comparison", "ok",
             f"previous={report.comparison.n_previous}, trend={report.comparison.trend}, "
             f"repeated_abnormal={report.comparison.repeated_abnormal}", t0)

        # 10. insights + diet/medication context ---------------------------
        t0 = time.perf_counter()
        notable = (cls.category not in ("firm, well formed", "soft but formed") or bool(flags)
                   or report.comparison.trend in ("looser", "firmer", "changed_color"))
        report.associations = (self.insights.associations(cls, report.observations, refs)
                               + self.diet.associations(pet, ts, notable, refs))
        step("diet_medication_context", "ok",
             f"notable={notable}; context associations="
             f"{sum(1 for a in report.associations if a.basis == 'history_context')}", t0)

        # 11. escalation ------------------------------------------------------
        t0 = time.perf_counter()
        report.escalation = self.escalation.evaluate(cls, report.observations, report.comparison, symptoms, refs, llm_flags)
        report.next_steps = self.insights.next_steps(cls, report.observations, report.comparison, report.escalation,
                                                      report.image_quality.warnings)
        step("escalation", "ok", f"level={report.escalation.level}; rules={report.escalation.rules_triggered or 'none'}", t0)

        # 12. safety validation ------------------------------------------------
        t0 = time.perf_counter()
        report.validation = self.validator.enforce(report)
        step("safety_validation", "ok" if report.validation.passed else "error",
             f"passed={report.validation.passed}; redactions={len(report.validation.redactions)}; issues={report.validation.issues or 'none'}", t0)

        # 13. persist ------------------------------------------------------------
        t0 = time.perf_counter()
        if persist and report.validation.passed and report.comparison.trend != "duplicate":
            self.history.add(pet, report.to_history_record())
            report.stored = True
            step("persist", "ok", f"stored in {pet.name}'s history ({len(pet.stool_records)} records)", t0)
        else:
            why = ("persistence disabled" if not persist else "duplicate photo" if report.comparison.trend == "duplicate"
                   else "validation failed")
            step("persist", "skipped", why, t0)
        trace.append(StepRecord("report", "ok", f"status={report.status}", 0.0))
        return report
