# AI Collaboration Reflection — TEMPLATE (please verify and personalise)

> Items marked **[SUGGESTED – verify]** describe what actually happened in the AI-assisted build (see `ai_interactions.md` Part 3). Keep, edit or delete them so this reflects *your* experience. Replace every `<...>` placeholder. Do not submit claims you cannot defend in your own words.

## 1. How AI assisted with system design
* **[SUGGESTED – verify]** The AI inspected the existing PawPal+ code first and extended it (new `pawpal_ai` package, one added `Pet.stool_records` field) instead of rewriting it; the original 11 tests still pass.
* **[SUGGESTED – verify]** It separated business logic from AI logic and made image input source-agnostic (`ImageInput` bytes + `source_type`).
* My own design decisions: `<what I accepted/changed, what I would draw differently>`

## 2. How AI assisted with coding and debugging
* **[SUGGESTED – verify]** Retrieval bug: the species filter ran after top-k truncation, so a cat-only passage hid the dog scoring chart. Fixed by filtering before ranking, with a regression test.
* **[SUGGESTED – verify]** Segmentation lost pale "mucus-like" patches on light tile; hole-filling added.
* **[SUGGESTED – verify]** A hard-coded `+= 0` in a first baseline comparison was caught in review and replaced by real measurement.
* What I ran/debugged myself: `<...>`

## 3. At least one AI suggestion I ACCEPTED
* `<SUGGESTED example: a rule-based escalation engine with listed rule IDs rather than letting a model decide urgency, so every escalation is auditable.>` Why: `<...>`

## 4. At least one AI suggestion I REJECTED or REVISED
* `<SUGGESTED example: a cat 1–7 score was rejected; no valid feline 1–7 scale was applied (a different 1–5 chart exists).>`
* `<SUGGESTED example: a single hard threshold for "normal" was revised into a probabilistic scorer with alternatives and low-confidence handling, since neighbouring scores are visually ambiguous.>`
* What I changed and why: `<...>`

## 5. Technical limitations (be candid)
* Heuristic features, not a trained model; **no real-photo validation**; synthetic same-author test data (results are circular).
* Optional LLM path never run against the live API; Streamlit tab only stub-tested in the build environment — `<note whether you ran them>`.
* Small, paraphrased, partly consumer-sourced knowledge base.

## 6. Future development
* Consented, veterinarian-labelled photos; train and calibrate a model; real held-out evaluation.
* Veterinarian review of knowledge base and escalation rules; feline scale after validation.
* Camera / litter-box adapters via the existing `source_type` hook.
* `<your ideas>`

## 7. What I learned / how my process went
`<in your own words: time spent, surprises, what you would do differently>`
