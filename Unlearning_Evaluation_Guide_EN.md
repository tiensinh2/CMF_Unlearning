# Evaluation Guide for Unlearning Experiment Results

This document defines the **correctness and quality criteria** the agent must
use when synthesizing, comparing, and interpreting results from the
experiment notebooks (NB1–NB5). Apply this guide EVERY TIME results are
aggregated or discussed — do not invent alternative criteria.

---

## 1. Three metrics that must always be reported together

For EVERY method, EVERY seed, EVERY split ratio, always report all three
metrics for both the retain-set and the forget-set:

| Metric | What it measures | How it's computed |
|---|---|---|
| **Output accuracy** | What the model actually predicts | Forward pass through the full model (encoder + its current classifier) |
| **Linear Probe accuracy** | How much discriminative information remains in the representation, via a FRESH classifier (trained from scratch) | Freeze the encoder, train a new classifier via cross-entropy on the full dataset, evaluate on forget/retain |
| **NCC accuracy** | How much discriminative information remains in the representation, purely geometrically (no training) | Freeze the encoder, compute class means, classify by nearest-class-center |

**Never** conclude "method X forgets well" based on Output accuracy alone —
this is the metric most easily "gamed" (see Section 3).

---

## 2. Determining the correct comparison target — THE MOST IMPORTANT STEP

**The most common mistake**: comparing forget accuracy to 0% in every case.
This is only correct for whole-class forgetting. Before evaluating any
number, identify which split type is in use:

### 2.1 Whole-class forgetting (forget = an entire class or classes)

- Correct target: **forget accuracy → 0%** (the model has never seen this
  class in any form, so the concept of "this class" doesn't exist in its
  decision boundary).
- Oracle Retrain's own forget accuracy in this setting is itself near 0% at
  the Output level (but NOT necessarily near 0% at the Probe/NCC level — see
  Section 2.3).

### 2.2 Sample-wise / random-mix forgetting (forget = a percentage of samples
### mixed across many classes, with every class still fully present in the
### retain set)

- Correct target: **forget accuracy → Oracle Retrain's Output/Probe/NCC
  accuracy on that SAME split** — NOT 0%.
- Reason: Oracle Retrain never saw the exact forgotten samples, but it did
  see thousands of OTHER samples from the same class in the retain set —
  it generalizes correctly to the forgotten samples due to within-class
  statistical similarity, not because it "remembers" them. Oracle's forget
  accuracy in this setting is therefore typically HIGH (e.g. ~0.90–0.95),
  not near 0%.
- **Always compute and report a "Δ vs Oracle" column** =
  `|forget_acc(method) − forget_acc(oracle)|` for every method, instead of
  reporting only the raw value.

### 2.3 General rule, applicable to both split types

Let `d(method) = forget_acc(method) − forget_acc(oracle)`.

- `d ≈ 0` (within ~1 standard deviation of Oracle's own across-seed
  variance): **GOOD — "genuine forget"**, the method behaves exactly like a
  model that never saw the forgotten data.
- `d > 0` and meaningfully large (method's forget acc is CLEARLY HIGHER than
  Oracle's): **BAD — "under-forgetting"**, the method hasn't really
  forgotten — it retains more information than even a model that was never
  trained on that data.
- `d < 0` and meaningfully large (method's forget acc is CLEARLY LOWER than
  Oracle's): **BAD — "over-forgetting"**, the method is destroying more than
  necessary — this is NOT a good outcome even though the raw number looks
  impressively low. This is usually accompanied by a retain-accuracy drop
  (see Section 4), since the mechanism causing over-forgetting typically
  spills over onto samples that should have been kept.

**Do not default to "lower forget accuracy is always better."** The best
outcome is the smallest absolute `d` (closest to Oracle), NOT the most
negative `d`.

---

## 3. Interpreting the Output vs Probe/NCC gap (the "illusion" flag)

For every method, always compute:

```
gap_probe = output_forget_acc − probe_forget_acc
gap_ncc   = output_forget_acc − ncc_forget_acc
```

(Output is usually lower than Probe/NCC — the gap is usually negative or
zero.)

| Situation | Interpretation | Label |
|---|---|---|
| `output_forget_acc` is low (near 0 or near Oracle) **AND** `probe_forget_acc`/`ncc_forget_acc` are also near Oracle | The model genuinely changed its representation — real forgetting | **GENUINE FORGETTING** |
| `output_forget_acc` is low **BUT** `probe_forget_acc`/`ncc_forget_acc` are much higher (close to the un-unlearned Original model) | Only the classifier was manipulated; the representation barely changed | **ILLUSION / MISALIGNMENT** — flag this explicitly in the report; do NOT count this as successful unlearning just because Output looks clean |
| `output_forget_acc` AND `probe_forget_acc`/`ncc_forget_acc` are BOTH clearly lower than Oracle | Not an illusion, but **over-forgetting** (see Section 2.3) | **OVER-FORGETTING** |

Always show both the Output value and the Probe/NCC value side by side in
every summary table — never report Output accuracy alone.

---

## 4. Evaluating retain accuracy — the "collateral damage" criterion

```
retain_drop = retain_acc(Original or Oracle) − retain_acc(method)
```

- Small `retain_drop` (within ~2–3 percentage points of Oracle): **GOOD** —
  the method preserves utility.
- Large `retain_drop`: **BAD — collateral damage**, no matter how good the
  forget accuracy looks. A method that severely degrades retain accuracy
  while trying to forget a small fraction of data is a failure, not an
  acceptable trade-off, unless an intentional trade-off is explicitly stated
  and justified in the report — never assume it's fine by default.

**Always report `retain_drop` alongside forget accuracy in the SAME row/table**
— never separate these two metrics when comparing methods, since a method
that "forgets well" but destroys retain accuracy is not a better method.

---

## 5. Standard summary table — always include these columns

For every (method, split_type, ratio, seed), the results table MUST include:

```
method | split_type | ratio | seed |
output_retain_acc | output_forget_acc |
probe_retain_acc  | probe_forget_acc  |
ncc_retain_acc    | ncc_forget_acc    |
oracle_output_forget_acc | oracle_probe_forget_acc | oracle_ncc_forget_acc |
d_output | d_probe | d_ncc |               # distance to Oracle (Section 2.3)
retain_drop_output | retain_drop_probe | retain_drop_ncc |   # Section 4
gap_probe | gap_ncc |                       # illusion flag (Section 3)
verdict                                     # see Section 6
```

## 6. How to auto-assign a "verdict" to each row

Apply the checks below in order, stopping at the first one that matches:

1. If `retain_drop` is large (above a threshold, e.g. 5 percentage points
   relative to Oracle) → **`COLLATERAL_DAMAGE`**
2. Otherwise, if `|d_probe|` or `|d_ncc|` is large (the method deviates far
   from Oracle at the representation level, in either direction):
   - if the deviation is toward higher-than-Oracle (`d > 0`) →
     **`UNDER_FORGETTING`**
   - if the deviation is toward lower-than-Oracle (`d < 0`) →
     **`OVER_FORGETTING`**
3. Otherwise, if `gap_probe`/`gap_ncc` is large (Output is low but
   Probe/NCC remain high, close to the Original model) → **`ILLUSION`**
4. If none of the above flags apply → **`GENUINE_FORGET`**

The specific threshold for "large" (how many percentage points) should be
derived from Oracle Retrain's own across-seed standard deviation — use e.g.
`1.5 × std` as a default threshold rather than a fixed absolute number for
every dataset.

---

## 7. When comparing MULTIPLE methods against each other

- Do NOT rank methods by Output accuracy alone.
- Do NOT rank methods by "lowest forget accuracy" (this may just be
  over-forgetting — see Section 2.3).
- Correct ranking: the "best" method in a group is the one with
  `verdict = GENUINE_FORGET` AND the smallest `retain_drop`. If no method in
  the group achieves `GENUINE_FORGET`, state this explicitly: "no method in
  this benchmark achieves true unlearning by both criteria" — this is a
  valid and useful conclusion; do not force-pick a "winner" that doesn't
  actually qualify.

## 8. When comparing the SAME method across split types (whole-class vs
## sample-wise)

This is the key comparison for the current research direction. For every
method, report whether its `verdict` CHANGES between the two split types:

- If a method achieves `GENUINE_FORGET` under whole-class but
  `OVER_FORGETTING` or `COLLATERAL_DAMAGE` under sample-wise → this method
  **does not generalize well** to sample-wise forgetting — state the
  suspected mechanism if known (e.g. relying on a single shared
  class-mean/anchor for both forget and retain samples within the same
  class).
- If the verdict stays the same across both splits → the method
  **generalizes well**.

Summarize this as a matrix: rows = method, columns = split type, cell value
= verdict — this is the core table for the comparative analysis section.

---

## 9. Things NOT to do when aggregating results

- Do not round away or omit standard deviation — always report mean ± std
  across seeds.
- Do not compare two methods run under different hyperparameters/epoch
  budgets as if they were on equal footing — always report the config
  (epochs, LR, mean_source, etc.) alongside each result.
- Do not use absolute language like "CMF is better" or "method X is wrong" —
  always qualify the claim with the specific condition ("under the
  whole-class split", "at the Output level but not the Probe level", etc.).
- Do not omit rows flagged `ILLUSION` when compiling the overall summary —
  these are usually the MOST important rows to highlight, not results to
  hide.
