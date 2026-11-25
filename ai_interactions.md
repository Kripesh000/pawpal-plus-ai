# AI Interactions Log

## Part 1 — Agent Workflow

### Task Given to the Agent
A single, specification-style prompt asked the agent to act as a lead architect and build PawPal+ end to end — core classes (`Task`, `Pet`, `Owner`, `Scheduler`), five algorithmic features (sorting, filtering, conflict detection, automated recurrence, next-available-slot finder), JSON persistence, a Streamlit UI, a `tabulate`/ANSI CLI demo, a `pytest` suite, a Mermaid UML diagram, and documentation — targeting all core requirements plus the five stretch challenges. It also specified exact file names, class/method names, test names, and a one-line docstring on every method.

### Files Created or Modified

| File | Action |
| :--- | :--- |
| `pawpal_system.py` | Created — domain model, `Scheduler`, `JsonPersistence` |
| `main.py` | Created — CLI demo |
| `tests/test_pawpal.py`, `conftest.py` | Created — 11 tests; root `conftest.py` makes imports work for plain `pytest` too |
| `app.py` | Replaced the starter placeholder with the full Streamlit UI |
| `diagrams/uml_final.mmd` | Created |
| `README.md`, `reflection.md`, `ai_interactions.md` | Rewritten from the starter templates |
| `requirements.txt`, `.gitignore` | Updated (`tabulate`, `wcwidth`; ignore `data.json`) |

### Agent Execution Steps
1. Inspected the uploaded starter repo (listed the zip, then read `app.py`, `README`, reflection template, `requirements`).
2. Checked the environment: `tabulate` was present; `pytest` and `streamlit` were not installed and there was no network access.
3. Implemented `pawpal_system.py` with validation, a sweep-line conflict detector, `timedelta`-based recurrence, and a gap-walking slot finder.
4. Built and ran `main.py`; reviewed the output by hand (e.g., confirmed the 07:30–08:00 walk overlaps the 07:45–08:00 breakfast by 15 min).
5. Wrote the test suite; because `pytest` was unavailable, ran it through a throwaway stand-in runner (`fixture`, `raises`, `tmp_path` only; not part of the repo).
6. Mutation-tested the suite with seven deliberately injected bugs.
7. Built `app.py`, then smoke-tested every UI path against a fake streamlit stub (since the real package wasn't installable).
8. Wrote the UML, `README` (embedding the actual captured CLI output), and reflection.

### Manual Corrections Made Along the Way
* **Test expectations were wrong, not the code.** Hand-tracing found five bad assertions (exact-fit gaps do fit; a 12:05–18:00 gap fits 250 min; Mochi has a 45-min gap at 07:00; a pending count was off by one; one conflict assertion indexed the wrong pair). Fixed before trusting any green result.
* **Tautological assertion removed.** A tautological assertion (`... == 3 + 0 or True`) slipped into a first draft of the conflict test and was replaced with a real assertion.
* **Misleading mutation result.** One injected bug appeared to fail the wrong tests. Root cause: Python reused stale `.pyc` bytecode because two mutated files were the same size and modified in the same second. Re-ran with `PYTHONDONTWRITEBYTECODE=1`; all seven bugs were then caught by the expected tests.
* **Emoji alignment.** `tabulate` mis-measures double-width emoji without `wcwidth`, breaking table borders, so emoji were moved out of CLI table cells (they remain in headings, messages and the Streamlit UI) and `wcwidth` was added to `requirements.txt`.
* **API tidy-ups.** `_pet_name` made public as `pet_name_for` because the CLI and UI both need it; a mid-function import in `app.py` moved to the top.

### What Was Not Verified (Be Aware)
* The sandbox had no `pytest`, so real `python -m pytest` output was not captured; the `README` has a marked block to paste yours into.
* The Streamlit app was exercised only against a stub, not the real `streamlit run`, so run it once and click through the tabs.
* The Mermaid file was syntax-checked structurally but not rendered with Mermaid.

---

## Part 2 — Model / Prompt Comparison: Recurring-Task Datetime Math

Status: template to complete. I have not run Claude 3.5 Sonnet or GPT-4o on this task, so no outputs are reported below. Fill in the "Observed" fields from real runs; the prompt, test cases, and rubric are ready to use.

### The Prompt (Identical for Both Models)
In Python, write `handle_recurring_task(task, pet)` for a pet-care app. `task` has `date` ("YYYY-MM-DD"), `time` ("HH:MM"), `frequency` ("Once" | "Daily" | "Weekly"), `description`, `priority`, `duration_mins`, `category`, `status`. When a Daily or Weekly task is completed, create and attach to the pet the next occurrence (same fields, status "Pending", new id, date advanced). Use `datetime.timedelta`. Return the new Task or None. Do not create duplicates.

### Shared Test Cases (Ground Truth from This Repo's Implementation)

| Input Date | Frequency | Expected Next Date |
| :--- | :--- | :--- |
| `2026-10-13` | `Daily` | `2026-10-14` |
| `2026-10-13` | `Weekly` | `2026-10-20` |
| `2026-10-31` | `Daily` | `2026-11-01` |
| `2026-12-31` | `Daily` | `2027-01-01` |
| `2028-02-28` | `Daily` | `2028-02-29` (leap year) |
| `2026-12-29` | `Weekly` | `2027-01-05` |
| any | `Once` | `None` |
| complete same task twice | `Daily` | second call returns `None`, no extra task |

### Evaluation

| Criterion | Claude 3.5 Sonnet | GPT-4o |
| :--- | :--- | :--- |
| Uses `timedelta` (not manual day+1 / month math)? | observed: | observed: |
| Correct on all 8 cases above? | observed: | observed: |
| Handles `Once` / unknown frequency without crashing? | observed: | observed: |
| Prevents duplicate / double-spawn? | observed: | observed: |
| Copies all fields, resets status, new id? | observed: | observed: |
| Date-string ↔ datetime conversion correct (`strptime`/`strftime`)? | observed: | observed: |
| Code clarity / docstring / length | observed: | observed: |
| Explained assumptions or edge cases unprompted? | observed: | observed: |

### Common Failure Modes to Look For
These are the typical ways this function goes wrong, and what the test cases above are designed to catch:
* Incrementing day + 1 on an integer or string → breaks at month/year end.
* Using `timedelta(months=1)` (doesn't exist) or approximating a week as `days=6` or `days=8`.
* Mutating the original task's date instead of creating a new task (loses history).
* Spawning from a task that is already complete → duplicates on repeated clicks.
* Forgetting to reset status to `"Pending"` on the new task, or sharing the old id.

### Useful Outputs / Flaws / Final Decision

| | Claude 3.5 Sonnet | GPT-4o |
| :--- | :--- | :--- |
| **Most useful part of the output** | observed: | observed: |
| **Flaws found** | observed: | observed: |
| **Changes I had to make** | observed: | observed: |

* **Final decision:** (fill in which output you based `handle_recurring_task` on and why; the version in `pawpal_system.py` uses `timedelta` via a `RECURRENCE_DELTAS` lookup table, copies fields into a fresh `Task`, and guards against duplicates.)

---

## Part 3 — AI-Assisted Development of PawPal+ AI (Project 4)

Record of what the coding agent actually did, from its working log. Edit if your own steps differed.

1. **Inspection:** cloned the GitHub repo (newer than the supplied zip), read `pawpal_system.py`, `app.py`, `README`; baseline suite: 11 passed.
2. **Research:** fetched sources to verify references (Purina 1-7 descriptions, Royal Canin feline 1-5 chart, AKC, Cornell, PetMD, Royal Canin colour guide). The Purina chart states no ideal score, so none is claimed; the feline chart is a different scale, so no numeric cat score is produced.
3. **Environment limits:** no network for installs, no real Streamlit, no real stool photos, no API key. Decisions: offline heuristic vision (documented as such), synthetic images, optional LLM path tested with a fake client only.
4. **Build order:** models, image processing/quality, synthetic samples, features/scoring (priors set after inspecting feature tables, checked on separate seeds), retrieval, history/diet/escalation, guardrails, pipeline, CLI/UI, tests, evaluation, docs.
5. **Defects found and fixed:** (a) "flat image" check wrongly rejected photos on grass; (b) segmentation lost pale patches matching the background (hole-filling added); (c) segmentation-band feature too noisy until its threshold was raised; (d) retrieval species filter applied after top-k (regression test added); (e) baseline counter was hard-coded to 0, replaced by real measurement; (f) feeding names were lower-cased in output.
6. **Harness validation:** six deliberate bugs injected; five caught, retrieval survived until RAG tests were added, then all six caught.
7. **Not verified:** live LLM calls; real Streamlit rendering; any real-world accuracy.

---

<!-- BEGIN RUNTIME TRACE (generated by `python -m pawpal_ai.cli demo`) -->
## Part 4 — Runtime Agentic Workflow Records (Generated)

Each analysis runs a fixed multi-stage workflow. The tables below are the concise, auditable execution records (stage, outcome, key values, duration) captured by the pipeline during `python -m pawpal_ai.cli demo`. They contain actions, validation results and decisions only; no private reasoning. Demo history for Luna is seeded and the images are synthetic.

### 1. Normal-Looking Stool (Dog, With Seeded History)
* Result: completed, score 3, category soft but formed, escalation none, safety validation passed

| Stage | Status | ms | Record |
| :--- | :--- | :---: | :--- |
| `input_validation` | ok | 10.6 | PNG 320x320, source=file_upload, species=dog |
| `quality_validation` | ok | 56.6 | score=1.0, warnings=none |
| `observation_extraction` | ok | 5.5 | method=heuristic-image-features-v1; flags=none; pieces=1 |
| `fecal_scoring` | ok | 0.1 | score=3, category=soft but formed, confidence=0.29 |
| `stability_check` | ok | 211.0 | agreement=0.75 across 4 perturbations |
| `reference_retrieval` | ok | 0.2 | 2 passages for topics ['scale', 'limits'] |
| `second_opinion` | skipped | 0.0 | no multimodal backend configured |
| `historical_comparison` | ok | 0.4 | previous=3, trend=looser, repeated_abnormal=False |
| `diet_medication_context` | ok | 0.2 | notable=True; context associations=2 |
| `escalation` | ok | 0.0 | level=none; rules=none |
| `safety_validation` | ok | 0.5 | passed=True; redactions=0; issues=none |
| `persist` | ok | 0.0 | stored in Luna's history (4 records) |
| `report` | ok | 0.0 | status=completed |

### 2. Loose Stool + Recent Diet Change (Dog)
* Result: completed, score 5, category loose / unformed, escalation monitor, safety validation passed

| Stage | Status | ms | Record |
| :--- | :--- | :---: | :--- |
| `input_validation` | ok | 3.4 | PNG 320x320, source=file_upload, species=dog |
| `quality_validation` | ok | 46.8 | score=1.0, warnings=none |
| `observation_extraction` | ok | 5.2 | method=heuristic-image-features-v1; flags=none; pieces=2 |
| `fecal_scoring` | ok | 0.0 | score=5, category=loose / unformed, confidence=0.44 |
| `stability_check` | ok | 203.5 | agreement=1.0 across 4 perturbations |
| `reference_retrieval` | ok | 0.3 | 4 passages for topics ['scale', 'limits', 'loose', 'diet'] |
| `second_opinion` | skipped | 0.0 | no multimodal backend configured |
| `historical_comparison` | ok | 0.1 | previous=4, trend=looser, repeated_abnormal=False |
| `diet_medication_context` | ok | 0.1 | notable=True; context associations=2 |
| `escalation` | ok | 0.0 | level=monitor; rules=['R12_loose_single', 'R14_trend_looser'] |
| `safety_validation` | ok | 0.5 | passed=True; redactions=0; issues=none |
| `persist` | ok | 0.0 | stored in Luna's history (5 records) |
| `report` | ok | 0.0 | status=completed |

### 3. Possible Red Discoloration, Schedule Vet Visit
* Result: completed, score 2, category firm, well formed, escalation contact_vet_soon, safety validation passed

| Stage | Status | ms | Record |
| :--- | :--- | :---: | :--- |
| `input_validation` | ok | 3.2 | PNG 320x320, source=file_upload, species=dog |
| `quality_validation` | ok | 45.6 | score=1.0, warnings=none |
| `observation_extraction` | ok | 2.8 | method=heuristic-image-features-v1; flags=['red_discoloration']; pieces=1 |
| `fecal_scoring` | ok | 0.1 | score=2, category=firm, well formed, confidence=0.35 |
| `stability_check` | ok | 204.0 | agreement=1.0 across 4 perturbations |
| `reference_retrieval` | ok | 0.3 | 3 passages for topics ['scale', 'limits', 'red'] |
| `second_opinion` | skipped | 0.0 | no multimodal backend configured |
| `historical_comparison` | ok | 0.1 | previous=5, trend=firmer, repeated_abnormal=False |
| `diet_medication_context` | ok | 0.1 | notable=True; context associations=2 |
| `escalation` | ok | 0.0 | level=contact_vet_soon; rules=['R1_blood_like_colour'] |
| `safety_validation` | ok | 0.5 | passed=True; redactions=0; issues=none |
| `persist` | ok | 0.0 | stored in Luna's history (6 records) |
| `report` | ok | 0.0 | status=completed |

### 4. Cat: Descriptive Category Only, No Numeric Score
* Result: completed, score None, category firm, well formed, escalation none, safety validation passed

| Stage | Status | ms | Record |
| :--- | :--- | :---: | :--- |
| `input_validation` | ok | 3.2 | PNG 320x320, source=file_upload, species=cat |
| `quality_validation` | ok | 49.1 | score=1.0, warnings=none |
| `observation_extraction` | ok | 3.0 | method=heuristic-image-features-v1; flags=none; pieces=1 |
| `fecal_scoring` | ok | 0.1 | score=None, category=firm, well formed, confidence=0.25 |
| `stability_check` | ok | 219.7 | agreement=1.0 across 4 perturbations |
| `reference_retrieval` | ok | 0.2 | 2 passages for topics ['scale', 'limits'] |
| `second_opinion` | skipped | 0.0 | no multimodal backend configured |
| `historical_comparison` | ok | 0.0 | previous=0, trend=no_history, repeated_abnormal=False |
| `diet_medication_context` | ok | 0.0 | notable=False; context associations=0 |
| `escalation` | ok | 0.0 | level=none; rules=none |
| `safety_validation` | ok | 0.3 | passed=True; redactions=0; issues=none |
| `persist` | ok | 0.0 | stored in Mochi's history (1 records) |
| `report` | ok | 0.0 | status=completed |

### 5. Repeated Loose Stool + Reported Symptoms
* Result: completed, score 6, category loose / unformed, escalation urgent, safety validation passed

| Stage | Status | ms | Record |
| :--- | :--- | :---: | :--- |
| `input_validation` | ok | 4.3 | PNG 320x320, source=file_upload, species=dog |
| `quality_validation` | ok | 57.7 | score=1.0, warnings=none |
| `observation_extraction` | ok | 3.3 | method=heuristic-image-features-v1; flags=['yellow_orange_stool']; pieces=1 |
| `fecal_scoring` | ok | 0.0 | score=6, category=loose / unformed, confidence=0.389 |
| `stability_check` | ok | 197.9 | agreement=1.0 across 4 perturbations |
| `reference_retrieval` | ok | 0.3 | 5 passages for topics ['scale', 'limits', 'loose', 'diet', 'yellow'] |
| `second_opinion` | skipped | 0.0 | no multimodal backend configured |
| `historical_comparison` | ok | 0.1 | previous=6, trend=looser, repeated_abnormal=True |
| `diet_medication_context` | ok | 0.1 | notable=True; context associations=2 |
| `escalation` | ok | 0.0 | level=urgent; rules=['R4_symptoms_plus_stool_change', 'R8_yellow_orange', 'R11_persistent_abnormal', 'R12_loose_single', 'R14_trend_looser'] |
| `safety_validation` | ok | 0.7 | passed=True; redactions=0; issues=none |
| `persist` | ok | 0.0 | stored in Luna's history (7 records) |
| `report` | ok | 0.0 | status=completed |

### 6. Blurry Photo (Rejected for Quality)
* Result: rejected_quality, escalation inconclusive, safety validation passed

| Stage | Status | ms | Record |
| :--- | :--- | :---: | :--- |
| `input_validation` | ok | 1.6 | PNG 320x320, source=file_upload, species=dog |
| `quality_validation` | stopped | 47.9 | issues=['out_of_focus'], score=0.72 |
| `report` | stopped | 0.0 | rejected_quality: no observations were generated |

### 7. Invalid File (Rejected as Input)
* Result: rejected_input, escalation inconclusive, safety validation passed

| Stage | Status | ms | Record |
| :--- | :--- | :---: | :--- |
| `input_validation` | stopped | 17.2 | unreadable: File is not a readable image (corrupt or not an image). |
| `report` | stopped | 0.0 | rejected_input: no observations were generated |
<!-- END RUNTIME TRACE -->
