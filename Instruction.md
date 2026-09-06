verify task:   
\# TASK: Full rebuild — fix all deviations from the paper, rerun every notebook  
\# from scratch with correct paper-faithful config, save checkpoints after  
\# every notebook, and implement TWO variants of CMF+dynamic-W as separate  
\# notebooks (NB4: post-hoc W calibration after full paper-faithful training;  
\# NB5: budget-shared two-stage where the last 10 epochs of Phase 1 are  
\# reallocated to Phase 2).

\#\# Why we're doing a full rerun

Previous notebook runs used a config that does not reliably match the paper  
(hyperparameters, CMF mean computation, evaluation protocol, and split logic  
had confirmed or suspected deviations — see Part A). Rather than patching  
individual results, we are rerunning the ENTIRE pipeline from scratch with a  
verified paper-faithful config, and saving every checkpoint this time so we  
never need to rerun again for any future analysis.

\#\# PART A — Fix all deviations from the paper BEFORE rerunning anything

Apply every fix below, then verify each with evidence (file/line/snippet or  
runtime test) before proceeding to Part B. This is fix-and-verify, not  
verify-only — we've already confirmed a rerun is needed, so apply fixes as  
you find issues.

\#\#\# A.1 — CMF update rule (Algorithm 1\) must default to paper-faithful config

  \- Default \`mean\_source="train"\`: both the global mean μ AND every class's  
    per-class mean m\_c are computed from the FULL dataset (retain \+ forget  
    combined), with no filtering, in every place \`recompute\_cmf()\` is called  
    for \`cmf\_static\` and any CMF-wrapped baseline.  
  \- \`mean\_source="retain"\` remains available ONLY as an explicitly-toggled  
    ablation, never the default, and must not affect any paper-faithful run.  
  \- Confirm the sample-level L2-normalize → per-class mean → global mean →  
    center → re-normalize sequence matches Algorithm 1's math. Keep the one  
    known documented deviation (sample-level normalize before averaging) and  
    make sure it's disclosed in a code comment — do not silently "fix" this  
    to match the paper's raw-feature averaging, since our own CMF variant  
    intentionally uses this; just document it clearly as a disclosed  
    deviation, not a bug.

\#\#\# A.2 — cmf\_static must exactly replicate Algorithm 2

  Per-epoch: recompute\_cmf → freeze W → update encoder only, for every one  
  of the paper's stated epoch budget (do not truncate). No per-round or  
  per-batch shortcuts.

\#\#\# A.3 — All non-CMF baselines (SCRUB, NegGrad+, Random-label, SalUn, SVD,  
\#\#\# UNSIR) must use paper Table 4 hyperparameters exactly

  Cross-check every method's ACTUAL effective learning rate, epoch count,  
  and method-specific hyperparameters (SVD's α\_r/α\_f, etc.) at runtime  
  (log \`args\` / \`optimizer.param\_groups\[0\]\['lr'\]\`, not just a default config  
  file value) against paper Table 4\. If our shell scripts / configs disagree  
  with the paper by 10-100× (a known risk flagged previously), USE THE  
  PAPER'S TABLE 4 VALUES as ground truth and fix the config to match — do  
  not silently keep the divergent script values. Document in a single  
  config file (e.g. \`paper\_hparams.py\`) every hyperparameter used, sourced  
  directly from Table 4, so it's auditable in one place going forward.

\#\#\# A.4 — NCC/NC3 weight extraction must correctly handle CMF models

  Whatever function computes NCC accuracy must extract \`model.CMFweights.  
  weight\` for CMF models (not a wrong \`nn.Linear\`/\`Identity()\` fallback).  
  Verify with a direct runtime check: the weight matrix our eval code uses  
  must be identical to \`model.CMFweights.weight\` fetched directly, for every  
  CMF-trained checkpoint.

\#\#\# A.5 — CMF geometry consistency in Probe/NCC feature extraction

  For CMF models, Probe/NCC evaluation must apply the same feature  
  transform the CMF classifier itself uses (\`ẑ \= normalize(normalize(f) −  
  μ)\` via \`model.\_preprocess\_feats\_for\_cmf()\`). For non-CMF baselines, use  
  raw features as the paper does — confirm the eval code branches correctly  
  by model type in both directions.

\#\#\# A.6 — Evaluation protocol (forget/retain split) must be IDENTICAL and  
\#\#\# correctly documented across every notebook

  Pick ONE split protocol for this full rerun (we've been using a fixed 3:7  
  and 1:9 stratified random-mix split across all classes — confirm this is  
  what we want going forward, distinct from the paper's original whole-class  
  split, and label results accordingly). Every notebook (1-5) must load the  
  SAME split files generated once in Notebook 1 — never regenerate or use a  
  different split function downstream. Verify no notebook silently applies  
  a different split (e.g. a leftover \`\_test\_split\_30\_70()\` inconsistent with  
  training partition).

\#\#\# A.7 — Runtime/dispatch bugs

  \- Guard \`torch.cuda.Event(enable\_timing=True)\` calls with  
    \`if device.type \== 'cuda':\` wherever this code path is actually invoked.  
  \- Fix or avoid the \`"grad\_descent"\` dispatch key silently routing to  
    \`unlearn\_naive\` (full NegGrad+) instead of a real gradient-descent-only  
    (retain-only fine-tune) implementation — audit whether we've used this  
    key; if we need pure fine-tuning, implement a correct standalone  
    function and use a distinct key for it.

\#\#\# A.8 — Notebook provenance

  Confirm/update the repo clone source in every notebook. If continuing to  
  use the \`tiensinh2\` fork, diff it against \`ycgao1/CMF\_Unlearning\`  
  (official) for every file touched by A.1-A.5 above and pull in any missing  
  fixes; if switching to official, pin a specific commit hash for  
  reproducibility and update all 4-5 notebooks' clone line consistently.

\#\# PART B — Full rerun, ALL notebooks, with checkpoint saving

Rerun every notebook from scratch under the fixed config from Part A. For  
EVERY notebook below, implement checkpoint saving so no notebook ever needs  
re-running again:

  \- Save to \`/kaggle/working/checkpoints/\` (persistent Kaggle output, not  
    ephemeral paths).  
  \- Every checkpoint dict must be self-describing: \`model\_state\_dict\`,  
    \`config\` (all hyperparameters/flags for that run), \`seed\`, \`metrics\`  
    (final Output/Probe/NCC retain+forget accuracy), and any method-specific  
    flags (\`mean\_source\`, \`w\_init\_mode\`, etc.) — loadable independently  
    without needing the original notebook's variables in scope.  
  \- Use a consistent, documented naming scheme from this point on (define it  
    once, in NB1, and reference it in every subsequent notebook):  
      \`{method}\_{variant}\_{mean\_source}\_seed{seed}.pt\`  
    e.g. \`scrub\_cmf\_static\_train\_seed0.pt\`,  
         \`neggrad\_plus\_nocmf\_seed1.pt\`,  
         \`theta\_o\_seed0.pt\`, \`oracle\_seed0.pt\`.

\#\#\# NB1 — \`01\_train\_full.ipynb\`  
  \- Train Θ\_o on the full training set, paper hyperparameters (A.3-derived  
    LR/epochs for pretraining), CosineAnnealingLR WITH the paper's 5-epoch  
    warmup (previously missing — add it).  
  \- Generate and save the forget/retain split files (3 seeds) for BOTH the  
    3:7 and 1:9 stratified random-mix ratios, clearly named  
    \`forget\_indices\_ratio{30|10}\_seed{seed}.json\`.  
  \- Save \`theta\_o\_seed{seed}.pt\` per seed, plus training logs.  
  \- Enforce early stopping (patience=50) per paper spec, and log whether it  
    triggered.  
  \- Clearly tag TEST\_MODE vs full-run outputs in filenames/metadata so they  
    are never confused (e.g. \`\_testmode\` suffix if TEST\_MODE=True).

\#\#\# NB2 — \`02\_retrain\_oracle.ipynb\`  
  \- Retrain from scratch on retain-set-only, per seed, per ratio (3:7 and  
    1:9), using paper hyperparameters (same as Θ\_o's pretraining config,  
    minus forgotten samples).  
  \- Fix the wall-clock-variable-scoping bug: confirm \`res\['wall\_clock\_  
    minutes'\]\` is correctly 0.0 when checkpoint-skip triggers and correctly  
    populated when actually trained — verify with both a skip-path and a  
    train-path test run.  
  \- Save \`oracle\_ratio{30|10}\_seed{seed}.pt\` per seed/ratio.  
  \- Evaluate using the SAME split protocol as NB1 (A.6) — no \`\_test\_split\_  
    30\_70()\` substitution.

\#\#\# NB3 — \`03\_unlearn\_no\_cmf.ipynb\`  
  \- Run all non-CMF baselines (SCRUB, NegGrad+, Random-label, SalUn, SVD,  
    UNSIR) with paper Table 4 hyperparameters (A.3), on both 3:7 and 1:9  
    splits, ≥3 seeds each.  
  \- Save one checkpoint per (method, ratio, seed):  
    \`{method}\_nocmf\_ratio{30|10}\_seed{seed}.pt\`, embedding computed  
    Output/Probe/NCC metrics in the checkpoint dict.  
  \- Fix the forget-class-identification bug noted previously (empty  
    \`unlearn\_class=\[\]\` breaking any internal logic reading \`args.  
    unlearn\_class\`) — confirm forget/retain membership is correctly derived  
    from the loaded split indices, not from a class-index list, throughout.

\#\#\# NB4 — \`04\_unlearn\_cmf.ipynb\` — cmf\_static AND cmf\_static+post-hoc-W

  Two methods in this notebook, both starting from the SAME cmf\_static run:

  \*\*4a. cmf\_static\*\* (paper Algorithm 2, unchanged): run to completion (full  
  paper epoch budget, e.g. 50 epochs), for base\_method in {scrub,  
  neggrad\_plus, random\_label, salun}, mean\_source="train" default (+  
  mean\_source="retain" ablation), both ratios, ≥3 seeds. Save  
  \`{base\_method}\_cmf\_static\_{mean\_source}\_ratio{30|10}\_seed{seed}.pt\`  
  with embedded Output/Probe/NCC metrics — this checkpoint is the SOLE  
  source of truth for cmf\_static results; nothing downstream should retrain  
  it.

  \*\*4b. cmf\_static \+ post-hoc W calibration\*\* ("cmf\_static\_posthoc"):  
LOAD the cmf\_static checkpoint from 4a (same base\_method/mean\_source/  
ratio/seed) — do NOT retrain Stage 1, reuse the finished encoder \+ CMF  
weights exactly as saved.  
FREEZE the entire encoder (all params, permanently, for this whole step).  
PROMOTE W from buffer to a proper trainable parameter (implement a clean  
\`CMFWeightsTrainable\` subclass or \`make\_trainable()\` method — do NOT use  
the fragile \`del \_buffers\['weight'\]\` hack; ensure resulting checkpoints  
remain loadable via \`load\_state\_dict()\` into a fresh CMFWeights instance).  
for epoch in 1..k\_posthoc (config: try k\_posthoc \= 2, 5, 10 as separate  
runs):  
    for each FULL epoch over retain\_loader (config: phase2\_data \=  
    "retain\_only" or "retain\_plus\_forget", run both):  
        loss \= CrossEntropy(W · z\_theta(x), y)   \# real gradient, no  
                                                   \# closed-form reset  
        backprop, update W only  
    log Output/Probe/NCC retain+forget accuracy this epoch  
   Since the encoder never changes here, Probe/NCC MUST equal the 4a  
    checkpoint's Probe/NCC exactly (log both values side by side as a sanity  
    check — if they differ, something is leaking gradient into the encoder,  
    which is a bug to catch immediately). Only Output-level accuracy should  
    change. Save  
    \`{base\_method}\_cmf\_static\_posthoc\_k{k\_posthoc}\_{phase2\_data}\_  
    {mean\_source}\_ratio{30|10}\_seed{seed}.pt\`.

  Output: a single pivoted table per (base\_method, ratio, mean\_source)  
  showing 4a vs 4b (each k\_posthoc/phase2\_data variant) side by side:  
  Output/Probe/NCC retain+forget accuracy, so the "does post-hoc W  
  calibration change Output accuracy while Probe/NCC stay frozen" comparison  
  is directly readable.

\#\#\# NB5 — \`05\_unlearn\_cmf\_budget\_shared.ipynb\` — two-stage with SHARED  
\#\#\# (not additional) epoch budget

  Same total epoch budget as cmf\_static (e.g. 50 epochs total, unchanged —  
  this is the key difference from NB4's 4b, which ADDS epochs on top of a  
  complete cmf\_static run). Here, the LAST \`k\_shared\` epochs (config: try  
  k\_shared \= 10, matching your "bỏ bớt 10 epochs cuối" request) are  
  reallocated from Phase 1 to Phase 2:  
STAGE 1 — epochs 1 to (50 \- k\_shared): run cmf\_static's exact per-epoch  
loop (recompute\_cmf → freeze W → update encoder), UNCHANGED, for  
(50 \- k\_shared) epochs instead of the full 50\.

STAGE 2 — epochs (50 \- k\_shared \+ 1\) to 50: freeze the ENTIRE encoder  
(permanently for the rest of this run), promote W to trainable (same  
clean promotion mechanism as 4b, not the old hack), and run k\_shared  
FULL epochs of real gradient descent on W (CrossEntropy on retain\_loader  
or retain\_plus\_forget per config), same as 4b's inner loop.

  Save checkpoints at BOTH the Stage-1-end transition point AND the final  
  Stage-2 end, with distinct filenames:  
  \`{base\_method}\_cmf\_budgetshared\_k{k\_shared}\_stage1end\_{mean\_source}\_  
  ratio{30|10}\_seed{seed}.pt\` and  
  \`{base\_method}\_cmf\_budgetshared\_k{k\_shared}\_final\_{mean\_source}\_  
  ratio{30|10}\_seed{seed}.pt\`.

  Run for base\_method=scrub first (priority), k\_shared ∈ {10} (extend to  
  {5, 15} for a small sweep if time allows), same ratios/seeds/mean\_source  
  options as NB4.

  Output: a table comparing NB5's Stage-1-end (= cmf\_static run for only  
  40 epochs, NOT directly comparable to NB4's full-50-epoch cmf\_static) vs  
  NB5's final (after the 10-epoch W-only Stage 2\) — this isolates the  
  question "if we sacrifice 10 epochs of encoder training for 10 epochs of  
  W training, within the SAME total budget, is that a net win?" — a  
  genuinely different question from NB4's "if we get 50 epochs of encoder  
  training PLUS extra epochs of W training for free, does W training help?"

  Also produce ONE combined comparison table across NB4 (4a, 4b) and NB5  
  (stage1end, final) for base\_method=scrub, so all four conditions  
  (cmf\_static-50ep, cmf\_static-50ep+extra-W, cmf\_static-40ep-only,  
  cmf\_static-40ep+10ep-W-shared-budget) are visible side by side against the  
  same Oracle Retrain reference.

\#\# Metrics and comparison convention (applies to all notebooks)

  \- Report Output, Linear-Probe, and NCC retain+forget accuracy for every  
    config, mean±std over seeds.  
  \- For BOTH 3:7 and 1:9 ratios: compare forget accuracy against the Oracle  
    Retrain's own forget accuracy (NOT against 0), since under random-mix  
    forgetting the oracle itself doesn't reach 0% due to same-class  
    generalization — document this convention explicitly in every results  
    table's caption.

\#\# Constraints

  \- Complete Part A fixes and verify each with evidence BEFORE starting any  
    Part B reruns — a rerun under an unverified config wastes compute.  
  \- NB4's 4b must reuse 4a's checkpoint, never retrain Stage 1 independently.  
  \- NB5 is fully independent of NB4 (different epoch allocation), but should  
    reuse the same split files, same base Θ\_o checkpoint from NB1, and the  
    same evaluation code.  
  \- Every notebook must be resumable: if it crashes partway, already-saved  
    checkpoints must not need to be regenerated (check-before-train logic  
    like NB2's, applied consistently to NB3/4/5).  
  \- Log the config actually used (mean\_source, hyperparameters, split ratio,  
    seed) inside every saved checkpoint, not just in a separate log file, so  
    a checkpoint alone is enough to know exactly how it was produced.

\#\# Deliverable

  \- Part A fix report (evidence for every item A.1-A.8)  
  \- 5 rerun notebooks (\`01\_train\_full.ipynb\` through  
    \`05\_unlearn\_cmf\_budget\_shared.ipynb\`) each saving self-describing  
    checkpoints for every run to \`/kaggle/working/checkpoints/\`  
  \- Per-notebook results CSVs \+ the NB4 (4a vs 4b) pivoted table \+ the NB5  
    (stage1end vs final) pivoted table \+ the combined 4-condition comparison  
    table for base\_method=scrub described above  
