"""
_validate_table4.py
Quick self-contained check: every value in paper_hparams.py against Table 4
of arXiv:2604.08271v1 (PDF lines 2307-2434).
Run: python _validate_table4.py
"""
import sys
import paper_hparams as p

results = []

def chk(name, got, want, cite):
    if isinstance(want, float):
        ok = abs(got - want) < 1e-12
    else:
        ok = (got == want)
    results.append((ok, name, got, want, cite))

# ── Original pre-training ──────────────────────────────────────────────
chk("pretrain.lr_init CIFAR-10",       p.PRETRAIN_RESNET_CIFAR["lr_init"],        1e-2,  "Table4 L2308")
chk("pretrain.epochs CIFAR-10",        p.PRETRAIN_RESNET_CIFAR["epochs"],         300,   "Table4 L2308")
chk("pretrain.batch CIFAR-10",         p.PRETRAIN_RESNET_CIFAR["batch_size"],     128,   "Table4 L2308")
chk("pretrain.momentum CIFAR-10",      p.PRETRAIN_RESNET_CIFAR["momentum"],       0.9,   "Table4 L2308")
chk("pretrain.weight_decay CIFAR-10",  p.PRETRAIN_RESNET_CIFAR["weight_decay"],   5e-4,  "Table4 L2308")
chk("pretrain.lr_init TinyImageNet",   p.PRETRAIN_RESNET_TINY["lr_init"],         5e-2,  "Table4 L2312")

# ── Oracle retrain ────────────────────────────────────────────────────
chk("oracle.epochs cifar10",           p.ORACLE_RETRAIN["cifar10"]["epochs"],     200,   "Table4 L2314")
chk("oracle.lr cifar10",               p.ORACLE_RETRAIN["cifar10"]["lr_init"],    1e-2,  "Table4 L2314")
chk("oracle.epochs tinyimagenet",      p.ORACLE_RETRAIN["tinyimagenet"]["epochs"],150,   "Table4 L2319")

# ── Epoch budgets ─────────────────────────────────────────────────────
epoch_rows = [
    ("random_label", 3, "L2330"), ("salun", 3, "L2339"),
    ("grad_ascent_descent", 3, "L2348"), ("scrub", 3, "L2357"),
    ("tarun", 3, "L2366"),
    ("random_label_CMF_RemoveFC", 4, "L2390"), ("salun_CMF_RemoveFC", 4, "L2399"),
    ("grad_ascent_descent_CMF_RemoveFC", 3, "L2408"),
    ("scrub_CMF_RemoveFC", 3, "L2417"), ("tarun_CMF_RemoveFC", 3, "L2426"),
]
for m, e, lno in epoch_rows:
    chk(f"epochs.{m}", p.UNLEARN_EPOCHS[m], e, f"Table4 {lno}")

# ── LRs CIFAR-10 ──────────────────────────────────────────────────────
lr10 = [
    ("random_label", 1e-4, "L2330"), ("salun", 1e-4, "L2339"),
    ("grad_ascent_descent", 1e-4, "L2348"), ("scrub", 1e-4, "L2357"),
    ("tarun", 5e-5, "L2366"),
    ("random_label_CMF_RemoveFC", 2e-3, "L2390"),
    ("salun_CMF_RemoveFC", 2e-3, "L2399"),
    ("grad_ascent_descent_CMF_RemoveFC", 1e-4, "L2408"),
    ("scrub_CMF_RemoveFC", 5e-3, "L2417"),
    ("tarun_CMF_RemoveFC", 5e-5, "L2426"),
]
for m, v, lno in lr10:
    chk(f"lr.{m} cifar10/single", p.UNLEARN_LR[m]["cifar10"]["single"], v, f"Table4 {lno}")

# ── LRs CIFAR-100 ─────────────────────────────────────────────────────
lr100 = [
    ("random_label", 3e-3, "L2333"), ("salun", 1e-3, "L2342"),
    ("grad_ascent_descent", 5e-3, "L2351"), ("scrub", 1e-3, "L2360"),
    ("tarun", 3e-5, "L2369"),
    ("random_label_CMF_RemoveFC", 2e-3, "L2393"),
    ("salun_CMF_RemoveFC", 2e-3, "L2402"),
    ("grad_ascent_descent_CMF_RemoveFC", 1e-4, "L2411"),
    ("scrub_CMF_RemoveFC", 5e-3, "L2420"),
    ("tarun_CMF_RemoveFC", 5e-5, "L2429"),
]
for m, v, lno in lr100:
    chk(f"lr.{m} cifar100/single", p.UNLEARN_LR[m]["cifar100"]["single"], v, f"Table4 {lno}")

# ── LRs TinyImageNet ──────────────────────────────────────────────────
lrtiny = [
    ("random_label", 5e-4, "L2336"), ("salun", 5e-4, "L2345"),
    ("grad_ascent_descent", 5e-4, "L2354"), ("scrub", 5e-3, "L2363"),
    ("tarun", 2e-5, "L2372"),
    ("random_label_CMF_RemoveFC", 1e-2, "L2396"),
    ("salun_CMF_RemoveFC", 1e-2, "L2405"),
    ("grad_ascent_descent_CMF_RemoveFC", 3e-5, "L2414"),
    ("scrub_CMF_RemoveFC", 1e-3, "L2423"),
    ("tarun_CMF_RemoveFC", 2e-5, "L2432"),
]
for m, v, lno in lrtiny:
    chk(f"lr.{m} tiny/single", p.UNLEARN_LR[m]["tinyimagenet"]["single"], v, f"Table4 {lno}")

# ── Batch sizes ───────────────────────────────────────────────────────
chk("batch.scrub",            p.UNLEARN_BATCH.get("scrub", 128),            64,  "Table4 L2357")
chk("batch.scrub_CMF",        p.UNLEARN_BATCH.get("scrub_CMF_RemoveFC",128), 64, "Table4 L2417")

# ── SVD ───────────────────────────────────────────────────────────────
chk("svd.alpha_r cifar10",  p.UNLEARN_EXTRA["SVD"]["SVD_alpha_r"],                  1000, "Table4 L2375")
chk("svd.alpha_f cifar10",  p.UNLEARN_EXTRA["SVD"]["SVD_alpha_f"],                  30,   "Table4 L2375")
chk("svd.samples cifar10",  p.UNLEARN_EXTRA["SVD"]["SVD_samples"],                  900,  "Table4 L2375")
chk("svd.alpha_r cifar100", p.UNLEARN_EXTRA["SVD"]["cifar100"]["SVD_alpha_r"],      1000, "Table4 L2380")
chk("svd.alpha_f cifar100", p.UNLEARN_EXTRA["SVD"]["cifar100"]["SVD_alpha_f"],      30,   "Table4 L2380")
chk("svd.alpha_r tiny",     p.UNLEARN_EXTRA["SVD"]["tinyimagenet"]["SVD_alpha_r"],  30,   "Table4 L2385")
chk("svd.alpha_f tiny",     p.UNLEARN_EXTRA["SVD"]["tinyimagenet"]["SVD_alpha_f"],  10,   "Table4 L2385")

# ── UNSIR impair_lr ───────────────────────────────────────────────────
chk("unsir.impair_lr cifar10",     p.UNLEARN_EXTRA["tarun"]["tarun_impair_lr"],               5e-5, "Table4 L2366")
chk("unsir.impair_lr cifar100",    p.UNLEARN_EXTRA["tarun"]["cifar100"]["tarun_impair_lr"],   3e-5, "Table4 L2369")
chk("unsir.impair_lr tiny",        p.UNLEARN_EXTRA["tarun"]["tinyimagenet"]["tarun_impair_lr"], 2e-5, "Table4 L2372")
chk("unsir_cmf.impair_lr cifar10", p.UNLEARN_EXTRA["tarun_CMF_RemoveFC"]["tarun_impair_lr"],  5e-5, "Table4 L2426")
chk("unsir_cmf.impair_lr cifar100",p.UNLEARN_EXTRA["tarun_CMF_RemoveFC"]["cifar100"]["tarun_impair_lr"], 5e-5, "Table4 L2429")
chk("unsir_cmf.impair_lr tiny",    p.UNLEARN_EXTRA["tarun_CMF_RemoveFC"]["tinyimagenet"]["tarun_impair_lr"], 2e-5, "Table4 L2432")

# ── SCRUB extras ─────────────────────────────────────────────────────
chk("scrub.msteps",     p.UNLEARN_EXTRA["scrub"]["scrub_msteps"],    2,  "Table4 L2357")
chk("scrub.sgda_bsz",   p.UNLEARN_EXTRA["scrub"]["scrub_sgda_bsz"],  64, "Table4 L2357")
chk("scrub_cmf.msteps", p.UNLEARN_EXTRA["scrub_CMF_RemoveFC"]["scrub_msteps"],   2,  "Table4 L2417")

# ── SalUn threshold, NegGrad clip ────────────────────────────────────
chk("salun.threshold",         p.UNLEARN_EXTRA["salun"]["salun_threshold"],             0.5, "Table4 L2339")
chk("salun_cmf.threshold",     p.UNLEARN_EXTRA["salun_CMF_RemoveFC"]["salun_threshold"],0.5, "Table4 L2399")
chk("neggrad.grad_clip",       p.UNLEARN_EXTRA["grad_ascent_descent"]["grad_norm_clip"],1.0, "Table4 L2348")
chk("neggrad_cmf.grad_clip",   p.UNLEARN_EXTRA["grad_ascent_descent_CMF_RemoveFC"]["grad_norm_clip"], 1.0, "Table4 L2408")

# ── Source tag ────────────────────────────────────────────────────────
chk("HPARAM_SOURCE_CURRENT", p.HPARAM_SOURCE_CURRENT, "table4", "meta")

# ── Results ───────────────────────────────────────────────────────────
n_total = len(results)
fails = [(n, g, w, c) for ok, n, g, w, c in results if not ok]
passes = n_total - len(fails)

print(f"Passed {passes}/{n_total}")
if fails:
    print("\nFAILURES:")
    for n, g, w, c in fails:
        print(f"  FAIL  {n:<45s}  got={g!r:<12}  want={w!r:<12}  [{c}]")
    sys.exit(1)
else:
    print("ALL PASS — codebase matches Table 4 exactly")
