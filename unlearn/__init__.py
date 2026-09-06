from .tools import maybe_eval_and_save
from .salun import salun_unlearn, get_salun_mask, salun_CMF_unlearn
from .goel import goel_last_unlearn
from .ssd import ssd_unlearn
from .scrub import scrub_unlearn, scrub_CMF_unlearn
from .tarun import tarun_unlearn, tarun_CMF_unlearn
from .naive import unlearn_naive, unlearn_naive_CMF
from .grad_descent_only import unlearn_grad_descent_only  # BUG FIX: proper retain-only fine-tune
from .random_label import random_label_unlearn, random_label_once, random_label_CMF_unlearn, random_label_once_CMF_unlearn, random_label_unlearn_iter_eval, random_label_CMF_unlearn_iter_eval
from .random_label_middle import random_label_unlearn_layer3
from .SVD import SVD_unlearn
from .nc_prune import prune
from .CMF import CMF_fine_tuing as CMF_FT
from .retrain import re_train
from .cmf_two_stage import cmf_two_stage_unlearn  # Part B: two-stage schedule

# NOTE: "grad_descent" previously mapped to unlearn_naive (NegGrad+, ascent+descent).
# This was a bug: any run with --unlearn-method grad_descent produced NegGrad+ results,
# not retain-only fine-tuning. Those runs must be rerun with the corrected mapping.
# Affected checkpoints: checkpoints/grad_descent/**
unlear_func = {
    "salun": salun_unlearn,
    "salun_CMF_RemoveFC": salun_CMF_unlearn,
    "scrub": scrub_unlearn,
    "scrub_CMF_RemoveFC": scrub_CMF_unlearn,
    "goel": goel_last_unlearn,
    "ssd": ssd_unlearn,
    "tarun": tarun_unlearn,
    "tarun_CMF_RemoveFC": tarun_CMF_unlearn,
    "grad_ascent_descent": unlearn_naive,
    "random_label": random_label_unlearn,
    "random_label_layer3": random_label_unlearn_layer3,
    "random_label_once": random_label_once,
    "random_label_iter_eval": random_label_unlearn_iter_eval,
    "random_label_CMF_RemoveFC_iter_eval": random_label_CMF_unlearn_iter_eval,
    "grad_descent": unlearn_grad_descent_only,  # FIXED: was unlearn_naive (wrong)
    "SVD": SVD_unlearn,
    "retrain": re_train,
    "prune": prune,
    "CMF_FT": CMF_FT,
    "grad_ascent_descent_CMF": unlearn_naive_CMF,
    "CMF_FT_RemoveFC": CMF_FT,
    "grad_ascent_descent_CMF_RemoveFC": unlearn_naive_CMF,
    "random_label_CMF_RemoveFC": random_label_CMF_unlearn,
    "random_label_once_CMF_RemoveFC": random_label_once_CMF_unlearn,
    # Part B: two-stage CMF unlearning
    "cmf_two_stage": cmf_two_stage_unlearn,
}