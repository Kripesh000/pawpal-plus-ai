"""Glue between the AI analysis and the existing PawPal+ scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from pawpal_system import Owner, Pet, Scheduler, Task

from .models import AnalysisReport

VET_TASK_PREFIX = "Vet consult: stool observation"


def schedule_vet_followup(owner: Owner, pet: Pet, report: AnalysisReport, on_date: Optional[str] = None,
                          duration_mins: int = 30) -> Optional[Task]:
    """Add a High-priority 'Vet' task in the next free slot when the report recommends a vet visit.

    'urgent' books today, 'contact_vet_soon' tomorrow. Returns the new Task, or None when no
    escalation applies, a follow-up already exists that day, or no free slot exists.
    This is an explicit, opt-in action; analyses never create tasks on their own.
    """
    if report.escalation is None or report.escalation.level not in ("contact_vet_soon", "urgent"):
        return None
    base = datetime.strptime(report.created_at[:10], "%Y-%m-%d").date()
    day = on_date or (base if report.escalation.level == "urgent" else base + timedelta(days=1)).strftime("%Y-%m-%d")
    if any(t.category == "Vet" and t.description.startswith(VET_TASK_PREFIX) and t.date == day and not t.is_complete
           for t in pet.tasks):
        return None
    slot = Scheduler(owner).find_next_available_slot(duration_mins, date=day)
    if slot is None:
        return None
    task = Task(f"{VET_TASK_PREFIX} ({report.escalation.level.replace('_', ' ')})", slot, day,
                category="Vet", priority="High", duration_mins=duration_mins)
    pet.add_task(task)
    return task
