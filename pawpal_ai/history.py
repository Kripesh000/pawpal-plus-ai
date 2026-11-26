"""Per-pet stool history stored on the PawPal+ Pet object, plus trend comparison."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pawpal_system import Pet

from .models import HistoricalComparison

CATEGORY_RANK = {"very hard / dry": 0, "firm, well formed": 1, "soft but formed": 2, "loose / unformed": 3, "watery / liquid": 4}
ABNORMAL_FLAGS = ("red_discoloration", "dark_or_black_stool", "pale_or_gray_stool", "yellow_orange_stool",
                  "mucus_like_coating", "foreign_material")
REPEAT_WINDOW_DAYS = 7


def is_abnormal_consistency(category: Optional[str]) -> bool:
    """True for categories outside 'firm, well formed' / 'soft but formed'."""
    return category in ("very hard / dry", "loose / unformed", "watery / liquid")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class PetHistoryManager:
    """Reads/writes `pet.stool_records` and compares a new observation with earlier ones."""

    @staticmethod
    def records(pet: Pet) -> List[Dict[str, Any]]:
        """All stored records for the pet, oldest first."""
        return sorted(pet.stool_records, key=lambda r: r["timestamp"])

    @staticmethod
    def add(pet: Pet, record: Dict[str, Any]) -> None:
        """Append one record (validated for the keys later code relies on)."""
        for key in ("report_id", "timestamp", "category", "flags", "escalation"):
            if key not in record:
                raise ValueError(f"history record is missing {key!r}")
        _parse(record["timestamp"])
        pet.stool_records.append(dict(record))

    def compare(self, pet: Pet, current: Dict[str, Any], now: datetime) -> HistoricalComparison:
        """Compare `current` (a history-record dict) with this pet's earlier records."""
        earlier = [r for r in self.records(pet) if _parse(r["timestamp"]) <= now and r["report_id"] != current["report_id"]]
        sha = current.get("image_sha256")
        dup = next((r for r in earlier if sha and r.get("image_sha256") == sha), None)
        if dup:
            return HistoricalComparison(
                n_previous=len(earlier),
                summary=f"This exact photo was already analysed on {dup['timestamp'][:10]}; "
                        "it is not counted as a new observation for trend purposes.",
                trend="duplicate",
            )
        if not earlier:
            return HistoricalComparison(
                n_previous=0, trend="no_history",
                summary="No earlier stool records for this pet, so no comparison is possible yet. "
                        "Trends become available after a few observations.",
            )

        recent = earlier[-5:]
        species = current.get("species", "dog")
        comp = HistoricalComparison(n_previous=len(earlier), summary="")
        parts: List[str] = []
        if species == "dog" and current.get("score") is not None:
            prior_scores = [r["score"] for r in recent if r.get("score") is not None]
            if prior_scores:
                ordered = sorted(prior_scores)
                median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
                comp.score_delta = round(current["score"] - median, 1)
                parts.append(f"Score {current['score']} vs typical recent score {median:g} (change {comp.score_delta:+g}).")
        else:
            prior_ranks = [CATEGORY_RANK[r["category"]] for r in recent if r.get("category") in CATEGORY_RANK]
            if prior_ranks and current.get("category") in CATEGORY_RANK:
                comp.score_delta = round(CATEGORY_RANK[current["category"]] - sorted(prior_ranks)[len(prior_ranks) // 2], 1)
                parts.append(f"Category '{current['category']}' (change vs typical recent: {comp.score_delta:+g} steps).")

        if comp.score_delta is not None:
            comp.trend = "looser" if comp.score_delta >= 1 else "firmer" if comp.score_delta <= -1 else "stable"

        colours = [r["color"] for r in recent if r.get("color")]
        if colours and current.get("color") and current["color"] != Counter(colours).most_common(1)[0][0]:
            parts.append(f"Colour appears {current['color']} compared with the usual {Counter(colours).most_common(1)[0][0]}.")
            if comp.trend in ("stable", "no_history"):
                comp.trend = "changed_color"

        window = [r for r in earlier if _parse(r["timestamp"]) >= now - timedelta(days=REPEAT_WINDOW_DAYS)]
        series = window + [current]
        abnormal = [r for r in series if is_abnormal_consistency(r.get("category"))]
        flag_counts = Counter(f for r in series for f in r.get("flags", []) if f in ABNORMAL_FLAGS)
        comp.repeated_flags = sorted(f for f, c in flag_counts.items() if c >= 2)
        comp.repeated_abnormal = len(abnormal) >= 2 or bool(comp.repeated_flags)
        if len(abnormal) >= 2:
            parts.append(f"{len(abnormal)} of the last {len(series)} observations in {REPEAT_WINDOW_DAYS} days were outside the firm/soft-formed range.")
        if comp.repeated_flags:
            parts.append("Repeated visual flags: " + ", ".join(comp.repeated_flags) + ".")
        if not parts:
            parts.append("Consistent with earlier observations.")
        comp.summary = " ".join(parts)
        return comp
