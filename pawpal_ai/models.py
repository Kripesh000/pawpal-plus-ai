"""Structured inputs and outputs shared by every stage of the stool-analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SOURCE_TYPES = ("file_upload", "smartphone", "litter_box", "monitoring_station")
SUPPORTED_SPECIES = ("dog", "cat")

# Escalation levels, ordered from least to most urgent. "inconclusive" = the photo could not be
# assessed, which is NOT the same as "none" (nothing concerning seen).
ESCALATION_LEVELS = ("inconclusive", "none", "monitor", "contact_vet_soon", "urgent")
ESCALATION_RANK = {level: i for i, level in enumerate(ESCALATION_LEVELS)}

# Finding statuses. "observed" = descriptive attribute; "possible" = flagged concern;
# "not_detected" never means "absent": a photo cannot prove absence.
FINDING_STATUSES = ("observed", "possible", "not_detected", "uncertain")


@dataclass
class ImageInput:
    """Capture-source-agnostic input: raw bytes plus where they came from."""

    data: bytes
    name: str = "image"
    source_type: str = "file_upload"
    captured_at: Optional[str] = None  # ISO timestamp if the capture device supplies one

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SOURCE_TYPES}")

    @classmethod
    def from_path(cls, path: str, source_type: str = "file_upload") -> "ImageInput":
        """Read an image file from disk."""
        import os

        with open(path, "rb") as handle:
            return cls(data=handle.read(), name=os.path.basename(path), source_type=source_type)


@dataclass
class ImageQualityReport:
    """Output 1: is the photo usable, and why or why not."""

    usable: bool
    score: float  # 0..1 heuristic quality score
    issues: List[str] = field(default_factory=list)  # blocking problems
    warnings: List[str] = field(default_factory=list)  # non-blocking problems
    metrics: Dict[str, float] = field(default_factory=dict)
    subject_visible: bool = False
    subject_confidence: float = 0.0
    recommendation: str = ""


@dataclass
class Finding:
    """One visual observation (what is visible), never a diagnosis."""

    name: str
    status: str  # one of FINDING_STATUSES
    confidence: float
    detail: str = ""


@dataclass
class VisualFeatures:
    """Numeric features measured from the image by the heuristic vision backend."""

    coverage: float = 0.0
    n_pieces: int = 0
    elongation: float = 0.0
    compactness: float = 0.0
    edge_roughness: float = 1.0
    segment_dips: int = 0
    texture: float = 0.0
    shine: float = 0.0
    mean_rgb: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    color_fractions: Dict[str, float] = field(default_factory=dict)
    pale_glossy_fraction: float = 0.0
    inclusion_fraction: float = 0.0


@dataclass
class FecalClassification:
    """Output 2: consistency category and (dogs only) the 1-7 score."""

    species: str
    scale: str  # human-readable scale name, or why none was used
    score: Optional[int]  # None for cats and for rejected images
    category: str
    description: str
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ConfidenceSummary:
    """Output 4: how much to trust the above, with the reasons."""

    overall: float
    level: str  # "high" | "moderate" | "low"
    components: Dict[str, float] = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)


@dataclass
class HistoricalComparison:
    """Output 5: how this observation compares with the pet's earlier records."""

    n_previous: int
    summary: str
    trend: str = "no_history"  # no_history | stable | firmer | looser | changed_color
    score_delta: Optional[float] = None
    repeated_abnormal: bool = False
    repeated_flags: List[str] = field(default_factory=list)
    consistency_check: Optional[Dict[str, Any]] = None


@dataclass
class Association:
    """Output 6: a *possible* link, always worded without claiming causation."""

    statement: str
    basis: str  # "visual_only" | "history_context"
    evidence_level: str = "possible_association"
    reference_ids: List[str] = field(default_factory=list)


@dataclass
class Escalation:
    """Output 8: whether and how urgently to involve a veterinarian."""

    level: str
    headline: str
    reasons: List[str] = field(default_factory=list)
    rules_triggered: List[str] = field(default_factory=list)
    reference_ids: List[str] = field(default_factory=list)


@dataclass
class RetrievedReference:
    """A knowledge-base passage retrieved for this analysis, with its citation."""

    ref_id: str
    title: str
    source: str
    url: str
    section: str
    snippet: str
    score: float


@dataclass
class ValidationResult:
    """Result of the structured-output and medical-safety validator."""

    passed: bool
    issues: List[str] = field(default_factory=list)
    redactions: List[str] = field(default_factory=list)


@dataclass
class StepRecord:
    """One auditable workflow step: what ran, its outcome, and a short decision note."""

    stage: str
    status: str  # "ok" | "warn" | "stopped" | "skipped" | "error"
    detail: str
    duration_ms: float = 0.0


@dataclass
class AnalysisReport:
    """Everything the pipeline returns. Sections map 1:1 to the eight required outputs."""

    report_id: str
    created_at: str
    pet_id: str
    pet_name: str
    species: str
    image_name: str
    source_type: str
    status: str  # "completed" | "rejected_input" | "rejected_quality"
    image_quality: Optional[ImageQualityReport] = None
    classification: Optional[FecalClassification] = None
    observations: List[Finding] = field(default_factory=list)
    confidence: Optional[ConfidenceSummary] = None
    comparison: Optional[HistoricalComparison] = None
    associations: List[Association] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    escalation: Optional[Escalation] = None
    references: List[RetrievedReference] = field(default_factory=list)
    disclaimers: List[str] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    trace: List[StepRecord] = field(default_factory=list)
    input_error: str = ""
    image_sha256: str = ""
    symptoms: List[str] = field(default_factory=list)
    stored: bool = False
    second_opinion: Optional[Dict[str, Any]] = None  # optional multimodal-LLM cross-check

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the whole report."""
        return asdict(self)

    def to_history_record(self) -> Dict[str, Any]:
        """Compact record stored in the pet's history (no raw image, no free text)."""
        cls = self.classification
        flags = [f.name for f in self.observations if f.status == "possible"]
        return {
            "report_id": self.report_id,
            "timestamp": self.created_at,
            "image_name": self.image_name,
            "image_sha256": self.image_sha256,
            "species": self.species,
            "score": cls.score if cls else None,
            "category": cls.category if cls else None,
            "color": next((f.detail for f in self.observations if f.name == "color"), None),
            "flags": flags,
            "confidence": self.confidence.overall if self.confidence else 0.0,
            "escalation": self.escalation.level if self.escalation else "none",
        }
