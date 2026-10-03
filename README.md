# clinical-eval-gate

**A release gate for diagnostic AI.** Evaluates an image classifier and the generated clinical report that
depends on it, then blocks the release when either regresses against a stored baseline.

Built on public data. Not a scoring script — a control that runs in CI and can say no.

---

## What is being evaluated

Two surfaces, evaluated together, because in a real diagnostic product they fail together.

**1. A multi-label condition classifier.** Given a chest radiograph, it emits a probability per condition.
Evaluated per condition, never in aggregate.

**2. A generated clinical report.** Given the classifier's output, a language model writes findings and a
conclusion in clinical prose. Evaluated for whether it says what the classifier actually found — and nothing
else.

The second depends on the first, so a regression in either can reach the reader as a wrong report. A harness
that checks only one of them is measuring half a system.

---

## What "good" means

Not accuracy. Four properties, each gated separately:

| Property | Question it answers | Gate |
|---|---|---|
| **Discrimination** | Can the model separate positive from negative for *this* condition? | Per-condition sensitivity and specificity against a per-condition floor |
| **Calibration** | When it says 0.9, is it right 90% of the time? | Brier score and a reliability curve; miscalibration blocks even when AUC holds |
| **Subgroup stability** | Does it hold across view, patient group, image quality? | Per-slice metrics; any slice below floor blocks |
| **Report faithfulness** | Does the prose claim only what the classifier found? | Zero tolerance — one invented finding fails the run |

The floors differ per condition, deliberately. A missed pneumothorax and a missed mild degenerative change
are not the same event, and a single threshold across all conditions encodes the false claim that they are.

---

## Why the naive metric is insufficient

This is the part of the project that matters. Five specific failures of the obvious approach:

### 1. Accuracy is a lie at low prevalence

A condition present in 2% of studies. A model that outputs "absent" unconditionally scores **98% accuracy**
and has zero clinical value. Accuracy rewards the base rate, not the model. Any metric that a constant
function can win is not a metric.

### 2. A single aggregate score hides the failure mode

An 88.5% pass rate tells you almost nothing. Scattered across accuracy, faithfulness and completeness, it
means the model is unsafe. Concentrated in one presentational dimension — say, the order findings are listed
in — it means the model is sound and needs a formatting fix.

Same number, opposite decisions. So every score in this harness decomposes by dimension and by condition
before it is allowed to inform a release decision. **The shape of a failure carries more information than its
rate.**

### 3. Discrimination without calibration is a trap

AUC measures ranking. It is indifferent to whether the probabilities mean anything. A model can rank cases
perfectly and still emit probabilities so compressed that no operating threshold gives useful behaviour —
and AUC will not tell you. Calibration is gated separately for this reason.

### 4. A median is not a summary when the spread is wide

Real published classifier suites report sensitivity ranging from the low 60s to 100 percent. A median in the
mid-70s describes none of those conditions. The number a clinician needs is the one for the condition in front
of them, so this harness reports per-condition and treats any aggregate as presentation, never as a gate input.

### 5. Text-similarity metrics do not measure clinical correctness

"No evidence of pneumothorax" and "Evidence of pneumothorax" differ by two characters and are clinically
opposite. BLEU and ROUGE score them as near-identical. They are useful for detecting *drift* between model
versions and useless for detecting *error*, so they are reported but never gate.

### And the one that applies to the harness itself

**An LLM judge is a model, not a ruler.** Rubric scores produced by a language model are one model's opinion
of another's output, and carry their own bias and variance. Any judge used here is validated against human
scoring on a stratified subset, with Cohen's kappa reported per dimension. A dimension scoring below 0.6
agreement is marked unvalidated and excluded from gating until its rubric is rewritten.

A harness that would not apply its own standards to itself has not earned the right to block anyone's release.

---

## The gate contract

On every pull request:

1. Run the evaluation suite against the golden set.
2. Compare each metric to the stored baseline.
3. Post a summary comment with per-condition and per-dimension deltas.
4. **Fail the build** when any gated metric falls below its floor, or regresses beyond tolerance.

The summary is designed so a reviewer never has to open a log. A gate that requires archaeology to interpret
is a gate people learn to override.

Baselines are versioned in-repo. Moving one is a reviewed commit with a stated reason — never a side effect
of a run.

---

## Data

Public datasets only. Nothing proprietary, nothing derived from prior employment.

- **Classifier evaluation** — an openly downloadable chest radiograph dataset with condition labels.
- **Report evaluation** — a dataset pairing studies with free-text radiology reports, which is what makes
  evaluating generated prose possible at all. These generally require credentialed access; the classifier
  half runs without it.

Every dataset ships with a **data card** recording provenance, label policy, known gaps, and observed
annotator disagreement. A golden set whose construction is undocumented is an unfalsifiable benchmark.

---

## Non-goals

Stated explicitly, because scope creep is how eval projects die:

- **Not a model training repo.** Models are inputs. This measures them.
- **Not a benchmark leaderboard.** The question is "did this change make it worse," not "which model wins."
- **Not a general-purpose eval framework.** It does one domain shape well rather than every shape poorly.
- **Not a clinical device.** Reference implementation for evaluation methodology. Nothing here is validated
  for clinical use.

---

## Status and limitations

Honest position, updated as the project moves:

- [ ] Golden set assembled and data card written
- [ ] Per-condition metrics with calibration
- [ ] Per-slice breakdown
- [ ] Report faithfulness checks (deterministic layer)
- [ ] Rubric-scored generative evaluation
- [ ] Judge validated against human scoring — kappa per dimension
- [ ] CI gate with baseline comparison and PR summary

**Known limitations**

- Judge validation uses a small human-scored subset; kappa confidence intervals are correspondingly wide.
- Slice definitions are chosen by hand and therefore encode assumptions about which subgroups matter.
- Calibration is measured on the golden set, which is not a random sample of deployment traffic.

---

## Design principle

**Cheap deterministic checks run before expensive probabilistic ones.**

A database or rule comparison that catches an entire class of regression costs nothing. Only what survives it
is worth spending an LLM judge on. Inverting that order is how evaluation bills grow without catching more.

---

## Running it

```bash
uv sync
uv run pytest                      # unit tests
uv run eval-gate run --config configs/baseline.yaml
uv run eval-gate compare --against baselines/v1.json
```

Python 3.12 · uv · pytest · ruff · pre-commit · GitHub Actions

---

## License

MIT
