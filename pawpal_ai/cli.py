"""Command-line interface: `python -m pawpal_ai.cli <analyze|demo|history|samples> ...`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from pawpal_system import Owner

from .demo import build_demo_owner, run_demo, write_execution_log
from .history import PetHistoryManager
from .integration import schedule_vet_followup
from .llm_backend import LLMVisionBackend
from .models import ImageInput
from .pipeline import StoolAnalysisPipeline
from .report import render_text


def _load_owner(path: Optional[str]) -> Owner:
    if path and os.path.exists(path):
        return Owner.load_from_json(path)
    return build_demo_owner()


def cmd_analyze(args: argparse.Namespace) -> int:
    owner = _load_owner(args.data)
    pet = owner.get_pet(args.pet)
    if pet is None:
        print(f"Unknown pet {args.pet!r}. Known pets: {[p.name for p in owner.pets]}", file=sys.stderr)
        return 2
    try:
        image = ImageInput.from_path(args.image, source_type=args.source)
    except OSError as exc:
        print(f"Cannot read image: {exc}", file=sys.stderr)
        return 2
    llm = None
    if args.llm:
        if LLMVisionBackend.is_configured():
            llm = LLMVisionBackend()
        else:
            print("Note: --llm requested but ANTHROPIC_API_KEY/anthropic SDK not available; continuing with heuristics only.", file=sys.stderr)
    symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()] if args.symptoms else []
    report = StoolAnalysisPipeline(llm=llm).analyze(pet, image, species=args.species, symptoms=symptoms, persist=not args.no_save)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(render_text(report, show_trace=args.trace))
    if args.schedule_vet:
        task = schedule_vet_followup(owner, pet, report)
        print(f"\nScheduler: {'added ' + repr(task) if task else 'no vet task added (no escalation, duplicate, or no free slot)'}")
    if args.data and not args.no_save:
        owner.save_to_json(args.data)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    runs = run_demo(verbose=not args.quiet)
    if args.log:
        write_execution_log(runs, args.log)
        print(f"\nExecution records written to {args.log}")
    for label, r in runs:
        print(f"{label:<60} -> {r.status:<17} escalation={r.escalation.level}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    owner = _load_owner(args.data)
    pet = owner.get_pet(args.pet)
    if pet is None:
        print(f"Unknown pet {args.pet!r}.", file=sys.stderr)
        return 2
    records = PetHistoryManager.records(pet)
    if not records:
        print(f"No stool records for {pet.name} yet.")
    for r in records:
        print(f"{r['timestamp']}  score={r.get('score')}  {r.get('category')}  colour={r.get('color')}  flags={r.get('flags')}  escalation={r.get('escalation')}")
    return 0


def cmd_samples(args: argparse.Namespace) -> int:
    from .samples import write_demo_images

    for name, path in write_demo_images(args.out).items():
        print(f"{name:<24} {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pawpal_ai", description="PawPal+ AI stool monitoring (not a diagnostic tool).")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="analyse one photo for one pet")
    a.add_argument("--image", required=True)
    a.add_argument("--pet", required=True, help="pet name or id")
    a.add_argument("--species", choices=["dog", "cat"], help="override the pet's species")
    a.add_argument("--data", help="PawPal+ JSON file to load/save (default: built-in demo owner, not saved)")
    a.add_argument("--symptoms", help="comma-separated: lethargy,vomiting,not_eating,not_drinking,pale_gums,straining,abdominal_pain")
    a.add_argument("--source", default="file_upload", choices=["file_upload", "smartphone", "litter_box", "monitoring_station"])
    a.add_argument("--schedule-vet", action="store_true", help="add a Vet task to the scheduler if escalation applies")
    a.add_argument("--llm", action="store_true", help="add an optional multimodal second opinion (needs ANTHROPIC_API_KEY)")
    a.add_argument("--no-save", action="store_true", help="do not store the result in the pet's history")
    a.add_argument("--trace", action="store_true", help="print the execution trace")
    a.add_argument("--json", action="store_true", help="print the full structured report as JSON")
    a.set_defaults(func=cmd_analyze)
    d = sub.add_parser("demo", help="run the seven built-in scenarios")
    d.add_argument("--log", help="write/refresh the runtime-trace section of this markdown file (e.g. ai_interactions.md)")
    d.add_argument("--quiet", action="store_true", help="only print the one-line summaries")
    d.set_defaults(func=cmd_demo)
    h = sub.add_parser("history", help="show a pet's stored stool records")
    h.add_argument("--pet", required=True)
    h.add_argument("--data")
    h.set_defaults(func=cmd_history)
    s = sub.add_parser("samples", help="regenerate the synthetic sample images")
    s.add_argument("--out", default="sample_images")
    s.set_defaults(func=cmd_samples)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
