# Model Card — PawPal+ AI stool observation prototype

## Summary
| | |
|---|---|
| Purpose | Help owners **monitor** stool appearance over time and decide when to contact a vet. **Not a diagnostic tool.** |
| Species | Dogs (Purina 1–7 estimate), cats (descriptive category only, no numeric scale) |
| Core method | Classical image processing: border-contrast segmentation → colour/shape/texture features → hand-set naive-Bayes-style priors per score, temperature-softened (T=3). **Not a trained CV model.** |
| Optional component | Multimodal LLM second opinion (`--llm`) with structured prompt + few-shot + JSON schema. **Never run against a live API in development**; only a fake client. |
| Training data | None. Priors are hand-set from the published Purina score descriptions and tuned by eye on synthetic images (seeds 1–8). |
| Evaluation data | 100% **synthetic** procedural images (`pawpal_ai/samples.py`), seeds ≥ 500 (held out from tuning). No real photographs. |

## Specialized behaviour (what makes it domain-specific)
* **Domain instructions** (`prompts.py` `SYSTEM_PROMPT`): describe only what is visible; no diagnosis, causation, medication/diet/fasting advice; decline when the image is unclear; allowed flag vocabulary only.
* **Constrained output formats:** a fixed 8-section `AnalysisReport` dataclass, and a seven-key JSON schema for model answers (`OUTPUT_SCHEMA`, enforced by `AIResponseValidator.validate_llm_payload`).
* **Few-shot examples** (`prompts.py` `FEW_SHOT`): three illustrative exemplars (normal, red streak, unusable image) written for this project — not real model outputs.
* **Retrieval:** per-topic BM25 retrieval of cited reference passages is passed into the prompt and attached to guidance items.

## Measured results (`python evaluate.py`, recorded run)
Total 81 · passed 81 · failed 0 · **100.0 %** · guardrail cases 55/55.
This is high because the thresholds were written by the same author as the synthetic generator; it shows the code does what its author intended on in-distribution inputs, **not** that it works on real stool photos. Harness sensitivity was checked by mutation: 6 injected bugs (quality gate off, safety filter off, escalation forced to "none", retrieval broken, history ignored, cats scored on the dog scale) were each caught by failing cases. (Retrieval initially survived, which led to adding the RAG test group.)

| Metric (synthetic, held-out seeds) | Result |
|---|---|
| Exact match to intended score (63 renders, 7 classes) | 56/63 (89 %) |
| Within ±1 of intended score | 63/63 (100 %) |
| Mean modal-class share across repeated renders | 0.89 (classes 3, 5, 6 lowest at 0.78) |
| Quality gate: unusable photos rejected with no observations | 8/8 |
| Invalid inputs rejected gracefully | 8/8 |
| Safety text filter: unsafe blocked / safe allowed | 8/8 and 6/6 (small hand-written set; will miss paraphrases) |

### Baseline vs specialized
"Baseline" = same feature extractor and scorer **without** the reliability layer (no quality gate, no safety filter, no retrieval; always emits a score). It is an ablation, not another product.

| Measure | Baseline | Full system |
|---|---|---|
| Score emitted for unusable-but-decodable images (blur, dark, empty, blue object) | 6/6 (wrongly) | 0/6 (all rejected, no observations) |
| Corrupt/empty files | 2/2 unhandled exceptions | 2/2 handled with a user-facing message |
| Escalation recall on 6 red/black-looking images | 6/6 | 6/6 (**no difference**; only graded levels, symptom and history rules differ) |
| Guidance items carrying a source citation (5 notable reports, 10 items) | 0/10 | 10/10 |
| Score accuracy | identical (shared scorer) | identical |

**Retrieval before/after** (`red_streaks.png`): association and escalation *wording* is identical with and without retrieval. Retrieval adds traceability: with it, both items cite `akc-blood-in-stool` (title, publisher, URL); without it they cite nothing. I make no claim that retrieval changes clinical content.

## Intended and out-of-scope use
Intended: tracking trends, prompting a vet conversation, organising photos with feeding/medication context. Out of scope: diagnosis, treatment/medication/diet decisions, emergencies (contact a vet directly), species other than dogs/cats, treating "none" as reassurance.

## Known failure modes and risks
* **False reassurance:** real problems may look normal to these features (microscopic blood, parasites, pale/yellow stool under odd light). Reports say a photo cannot rule problems out.
* **False alarms:** food dyes, supplements, reflections and lighting can trigger red/dark/pale/mucus flags.
* **Not a stool detector:** the quality gate only checks for a stool-*coloured* subject; other brown objects pass it.
* **Distribution shift:** real photos (clutter, grass, shadows, litter) are far outside the synthetic distribution; expect poorer segmentation, more rejections, unreliable scores. Confidence is not calibrated against real outcomes.
* **Reference quality:** several sources are consumer articles (flagged in the notes); none veterinarian-reviewed by me.

## Sources used for the knowledge base
Purina 1–7 scale (hosted by Univ. of Missouri VHC), Royal Canin feline 1–5 chart, AKC blood-in-stool article, Royal Canin colour overview, PetMD mucus article (J. Coates DVM; reviewer V. Higgs DVM), Cornell Riney Canine Health Center diarrhea article. Notes are **paraphrased summaries**; verify against originals.
