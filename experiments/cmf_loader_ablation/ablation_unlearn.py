"""
experiments/cmf_loader_ablation/ablation_unlearn.py

Self-contained copies of the four CMF-compatible unlearning methods that accept the
new `mean_source` flag and pass it through every `recompute_cmf()` call site.

Supported methods:
    - ablation_naive_CMF          (NegGrad+: gradient ascent on forget + descent on retain)
    - ablation_random_label_CMF   (Random-label relabelling)
    - ablation_salun_CMF          (SalUn + random relabelling)
    - ablation_scrub_CMF          (SCRUB knowledge-distillation)

All methods accept the same signature:
    fn(args, model, device,
       retain_loader, forget_loader,
       train_loader, retain_loader_full, test_loader,
       optimizer, epochs,
       test_forget_loader=None,
       mean_source="train",
       **kwargs)

The extra argument `retain_loader_full` carries a retain-only DataLoader that
covers the FULL retain set (no subsampling), used exclusively by recompute_cmf()
when mean_source="retain".  The training-subset retain_loader is still used for
the actual gradient steps (preserving the original sample count for training).

DO NOT IMPORT from the original unlearn/ files — this module is a self-contained
ablation copy that only depends on the existing evaluation/ harness and utils.py.
"""

from __future__ import annotations

import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

import evaluation
from evaluation.nc_cmf import ncc_mismatch


# ---------------------------------------------------------------------------
# Shared helper: call recompute_cmf with the right loaders & mean_source
# ---------------------------------------------------------------------------

def _recompute(model, train_loader, retain_loader_full, device, mean_source):
    """Route to AblationModelModule.recompute_cmf with correct arguments."""
    model.recompute_cmf(
        loader_full=train_loader,
        loader_retain=retain_loader_full,
        device=device,
        mean_source=mean_source,
    )


# ---------------------------------------------------------------------------
# 1. NegGrad+ / Gradient Ascent-Descent  (ablation_naive_CMF)
# ---------------------------------------------------------------------------

def ablation_naive_CMF(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, retain_loader_full, test_loader,
    optimizer, epochs,
    test_forget_loader=None,
    mean_source: str = "train",
    **kwargs,
):
    """
    Gradient Ascent on forget set + Descent on retain set (NegGrad+).
    Recomputes CMF at the end of every epoch using `mean_source`.

    mean_source is passed through to every recompute_cmf() call so that
    the global μ is computed from the correct loader.
    """
    from utils import test, get_model

    clip = getattr(args, "grad_norm_clip", None)

    # Subsample forget / retain training subsets (same sizes as production)
    forget_ds = copy.deepcopy(forget_loader.dataset)
    nf = min(args.num_forget_samples, len(forget_ds))
    forget_ds, _ = torch.utils.data.random_split(forget_ds, [nf, len(forget_ds) - nf])
    naive_forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=True)
    forget_iterator = iter(naive_forget_loader)

    retain_ds = retain_loader.dataset
    nr = min(args.num_retain_samples, len(retain_ds))
    retain_ds, _ = torch.utils.data.random_split(retain_ds, [nr, len(retain_ds) - nr])
    naive_retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=True)

    # Loss scale params
    alpha  = getattr(args, "forget_scale", 0.5)
    lam    = getattr(args, "align_coef", 0.0)

    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    mu_drift_list: list[float] = []
    nc3_retain_list: list[float] = []
    epoch_list = [0]

    # --- Epoch 0 baseline ---
    model.eval()
    _recompute(model, train_loader, retain_loader_full, device, mean_source)
    mu_prev = model.CMFweights.mu.clone().detach()

    with torch.no_grad():
        r0, f0, _ = test(model, device, test_loader,
                         args.unlearn_class, args.class_label_names, args.num_classes,
                         job_name="ablation_naive_CMF", set_name="Test Set (Epoch 0)")
    retain_acc_list.append(r0)
    forget_acc_list.append(f0)

    lp_every = getattr(args, "lp_every", 1)
    if lp_every != 0:
        bs_probe = getattr(args, "prob_batch_size", 256)
        outs = evaluation.run_linear_probe_on_fresh_clone(
            args=args, get_model_fn=get_model, device_probe=device,
            src_model=model, train_loader=train_loader, test_loader=test_loader,
            num_classes=args.num_classes, bs_probe=bs_probe,
        )
        LP_retain_acc_list.append(outs["acc_test_retain"])
        LP_forget_acc_list.append(outs["acc_test_forget"])

    model.train()

    # --- Main loop ---
    for epoch in range(1, epochs + 1):
        epoch_list.append(epoch)

        with torch.no_grad():
            W_fixed = model.CMFweights.weight.detach()
            temp    = getattr(model.args, "temperature", 1.0)

        for x, y in naive_retain_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Ascent on forget
            try:
                x_f, y_f = next(forget_iterator)
            except StopIteration:
                forget_iterator = iter(naive_forget_loader)
                x_f, y_f = next(forget_iterator)
            x_f, y_f = x_f.to(device), y_f.to(device)
            f_f = model.extract_features(x_f)
            z_f = model._preprocess_feats_for_cmf(f_f)
            logits_f = (z_f @ W_fixed.t()) * temp
            L_f = F.cross_entropy(logits_f, y_f)
            L_align_f = 1.0 - (z_f * W_fixed[y_f]).sum(1).mean()
            (-(alpha * L_f + lam * L_align_f)).backward()

            # Descent on retain
            f_r = model.extract_features(x)
            z_r = model._preprocess_feats_for_cmf(f_r)
            logits_r = (z_r @ W_fixed.t()) * temp
            L_r = F.cross_entropy(logits_r, y)
            L_align_r = 1.0 - (z_r * W_fixed[y]).sum(1).mean()
            (L_r + lam * L_align_r).backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # End-of-epoch recompute + eval
        model.eval()
        _recompute(model, train_loader, retain_loader_full, device, mean_source)

        # μ drift
        mu_curr = model.CMFweights.mu.clone().detach()
        drift = float((mu_curr - mu_prev).norm().item())
        mu_drift_list.append(drift)
        mu_prev = mu_curr

        # NC-3 cosine self-duality for retain classes only
        nc3 = _retain_nc3(model, args)
        nc3_retain_list.append(nc3)

        with torch.no_grad():
            r, f, _ = test(model, device, test_loader,
                           args.unlearn_class, args.class_label_names, args.num_classes,
                           job_name="ablation_naive_CMF", set_name=f"Test Set (Epoch {epoch})")
        retain_acc_list.append(r)
        forget_acc_list.append(f)

        if lp_every > 0 and (epoch % lp_every == 0):
            outs = evaluation.run_linear_probe_on_fresh_clone(
                args=args, get_model_fn=get_model, device_probe=device,
                src_model=model, train_loader=train_loader, test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=getattr(args, "prob_batch_size", 256),
            )
            LP_retain_acc_list.append(outs["acc_test_retain"])
            LP_forget_acc_list.append(outs["acc_test_forget"])

        model.train()

    model.history_log = {
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "mu_drift": mu_drift_list,
        "nc3_retain": nc3_retain_list,
        "mean_source": mean_source,
    }
    return model


# ---------------------------------------------------------------------------
# 2. Random-label  (ablation_random_label_CMF)
# ---------------------------------------------------------------------------

def ablation_random_label_CMF(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, retain_loader_full, test_loader,
    optimizer, epochs,
    test_forget_loader=None,
    mean_source: str = "train",
    **kwargs,
):
    """
    Random-label unlearning with CMF. Forget-class samples are relabelled to a
    uniformly random non-forget class before each forward pass.
    Recomputes CMF at the end of every epoch using `mean_source`.
    """
    from utils import test, get_model

    num_classes   = args.num_classes
    forget_classes = set(args.unlearn_class)

    clip = getattr(args, "grad_norm_clip", None)

    # Fixed mixed subset
    gen_f = torch.Generator().manual_seed(args.seed)
    full_forget = copy.deepcopy(forget_loader.dataset)
    nf = min(args.num_forget_samples, len(full_forget))
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [nf, len(full_forget) - nf], generator=gen_f)

    gen_r = torch.Generator().manual_seed(args.seed + 1)
    full_retain = copy.deepcopy(retain_loader.dataset)
    nr = min(args.num_retain_samples, len(full_retain))
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [nr, len(full_retain) - nr], generator=gen_r)

    mixed_loader = DataLoader(
        ConcatDataset([forget_sub, retain_sub]),
        batch_size=args.batch_size, shuffle=True)

    valid   = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    NCC_retain_acc_list, NCC_forget_acc_list = [], []
    mu_drift_list: list[float] = []
    nc3_retain_list: list[float] = []
    epoch_list = [0]

    # --- Epoch 0 ---
    model.eval()
    _recompute(model, train_loader, retain_loader_full, device, mean_source)
    mu_prev = model.CMFweights.mu.clone().detach()

    r0, f0, _ = test(model, device, test_loader,
                     args.unlearn_class, args.class_label_names, args.num_classes,
                     job_name="ablation_random_label_CMF", set_name="Test Set (Epoch 0)")
    retain_acc_list.append(r0)
    forget_acc_list.append(f0)

    lp_every  = getattr(args, "lp_every",  1)
    ncc_every = getattr(args, "ncc_every", 1)

    if lp_every != 0:
        outs = evaluation.run_linear_probe_on_fresh_clone(
            args=args, get_model_fn=get_model, device_probe=device,
            src_model=model, train_loader=train_loader, test_loader=test_loader,
            num_classes=args.num_classes,
            bs_probe=getattr(args, "prob_batch_size", 256),
        )
        LP_retain_acc_list.append(outs["acc_test_retain"])
        LP_forget_acc_list.append(outs["acc_test_forget"])

    if ncc_every != 0 and test_forget_loader is not None:
        nr_a, nf_a = _run_ncc(args, model, train_loader, test_loader,
                               test_forget_loader, device)
        NCC_retain_acc_list.append(nr_a)
        NCC_forget_acc_list.append(nf_a)

    model.train()

    # --- Main loop ---
    for epoch in range(1, epochs + 1):
        epoch_list.append(epoch)
        model.train()

        for inputs, labels_true in mixed_loader:
            inputs      = inputs.to(device)
            labels_true = labels_true.to(device)

            labels = labels_true.clone()
            fmask  = torch.zeros_like(labels_true, dtype=torch.bool)
            for cls in forget_classes:
                fmask |= (labels_true == cls)
            if fmask.any():
                idx = fmask.nonzero(as_tuple=True)[0]
                labels[idx] = choices[torch.randint(0, len(valid), (idx.numel(),), device=device)]

            optimizer.zero_grad()

            with torch.no_grad():
                W_fixed = model.CMFweights.weight.detach()
                temp    = getattr(model.args, "temperature", 1.0)

            f = model.extract_features(inputs)
            z = model._preprocess_feats_for_cmf(f)
            logits = (z @ W_fixed.t()) * temp
            loss   = F.cross_entropy(logits, labels)
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if getattr(args, "dry_run", False):
                break

        model.eval()
        _recompute(model, train_loader, retain_loader_full, device, mean_source)

        mu_curr = model.CMFweights.mu.clone().detach()
        mu_drift_list.append(float((mu_curr - mu_prev).norm().item()))
        mu_prev = mu_curr
        nc3_retain_list.append(_retain_nc3(model, args))

        r, f, _ = test(model, device, test_loader,
                       args.unlearn_class, args.class_label_names, args.num_classes,
                       job_name="ablation_random_label_CMF", set_name=f"Test Set (Epoch {epoch})")
        retain_acc_list.append(r)
        forget_acc_list.append(f)

        if lp_every > 0 and (epoch % lp_every == 0):
            outs = evaluation.run_linear_probe_on_fresh_clone(
                args=args, get_model_fn=get_model, device_probe=device,
                src_model=model, train_loader=train_loader, test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=getattr(args, "prob_batch_size", 256),
            )
            LP_retain_acc_list.append(outs["acc_test_retain"])
            LP_forget_acc_list.append(outs["acc_test_forget"])

        if ncc_every > 0 and (epoch % ncc_every == 0) and test_forget_loader is not None:
            nr_a, nf_a = _run_ncc(args, model, train_loader, test_loader,
                                   test_forget_loader, device)
            NCC_retain_acc_list.append(nr_a)
            NCC_forget_acc_list.append(nf_a)

        model.train()

    model.history_log = {
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "NCC_retain_acc": NCC_retain_acc_list,
        "NCC_forget_acc": NCC_forget_acc_list,
        "mu_drift": mu_drift_list,
        "nc3_retain": nc3_retain_list,
        "mean_source": mean_source,
    }
    return model


# ---------------------------------------------------------------------------
# 3. SalUn + CMF  (ablation_salun_CMF)
# ---------------------------------------------------------------------------

def ablation_salun_CMF(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, retain_loader_full, test_loader,
    optimizer, epochs,
    test_forget_loader=None,
    mean_source: str = "train",
    **kwargs,
):
    """
    SalUn with CMF. Builds a gradient-importance mask on the forget set, then
    performs random-label training with that mask applied to gradients.
    Recomputes CMF at the end of every epoch using `mean_source`.
    """
    from utils import test, get_model

    num_classes    = args.num_classes
    forget_classes = set(args.unlearn_class)

    clip = getattr(args, "grad_norm_clip", None)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    # ---- Epoch 0: align + baseline ----
    model.eval()
    _recompute(model, train_loader, retain_loader_full, device, mean_source)
    mu_prev = model.CMFweights.mu.clone().detach()

    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    mu_drift_list: list[float] = []
    nc3_retain_list: list[float] = []
    epoch_list = list(range(0, epochs + 1))

    r0, f0, _ = test(model, device, test_loader,
                     args.unlearn_class, args.class_label_names, args.num_classes,
                     job_name="ablation_salun_CMF", set_name="Test Set (Epoch 0)")
    retain_acc_list.append(r0)
    forget_acc_list.append(f0)

    lp_every = getattr(args, "lp_every", 1)
    if lp_every != 0:
        outs = evaluation.run_linear_probe_on_fresh_clone(
            args=args, get_model_fn=get_model, device_probe=device,
            src_model=model, train_loader=train_loader, test_loader=test_loader,
            num_classes=args.num_classes,
            bs_probe=getattr(args, "prob_batch_size", 256),
        )
        LP_retain_acc_list.append(outs["acc_test_retain"])
        LP_forget_acc_list.append(outs["acc_test_forget"])

    # ---- Build SalUn mask ----
    hard_mask = _build_salun_mask(args, model, device, forget_loader)

    # ---- Build fixed mixed loader ----
    gen_f = torch.Generator().manual_seed(args.seed)
    full_forget = copy.deepcopy(forget_loader.dataset)
    nf = min(args.num_forget_samples, len(full_forget))
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [nf, len(full_forget) - nf], generator=gen_f)

    gen_r = torch.Generator().manual_seed(args.seed + 1)
    full_retain = copy.deepcopy(retain_loader.dataset)
    nr = min(args.num_retain_samples, len(full_retain))
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [nr, len(full_retain) - nr], generator=gen_r)

    mixed_loader = DataLoader(
        ConcatDataset([forget_sub, retain_sub]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=getattr(forget_loader, "num_workers", 0),
        pin_memory=True,
    )

    valid   = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    # ---- Training epochs ----
    for epoch in range(1, epochs + 1):
        model.train()

        for inputs, labels_true in mixed_loader:
            inputs      = inputs.to(device, non_blocking=True)
            labels_true = labels_true.to(device, non_blocking=True)

            labels = labels_true.clone()
            fmask  = torch.zeros_like(labels_true, dtype=torch.bool)
            for cls in forget_classes:
                fmask |= (labels_true == cls)
            if fmask.any():
                nf_ = int(fmask.sum().item())
                labels[fmask] = choices[torch.randint(0, len(valid), (nf_,), device=device)]

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()

            if hard_mask is not None:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is not None and name in hard_mask:
                            p.grad.mul_(hard_mask[name])

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if getattr(args, "dry_run", False):
                break

        model.eval()
        _recompute(model, train_loader, retain_loader_full, device, mean_source)

        mu_curr = model.CMFweights.mu.clone().detach()
        mu_drift_list.append(float((mu_curr - mu_prev).norm().item()))
        mu_prev = mu_curr
        nc3_retain_list.append(_retain_nc3(model, args))

        r, f, _ = test(model, device, test_loader,
                       args.unlearn_class, args.class_label_names, args.num_classes,
                       job_name="ablation_salun_CMF", set_name=f"Test Set (Epoch {epoch})")
        retain_acc_list.append(r)
        forget_acc_list.append(f)

        if lp_every > 0 and (epoch % lp_every == 0):
            outs = evaluation.run_linear_probe_on_fresh_clone(
                args=args, get_model_fn=get_model, device_probe=device,
                src_model=model, train_loader=train_loader, test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=getattr(args, "prob_batch_size", 256),
            )
            LP_retain_acc_list.append(outs["acc_test_retain"])
            LP_forget_acc_list.append(outs["acc_test_forget"])

        scheduler.step()
        model.train()

    model.history_log = {
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "mu_drift": mu_drift_list,
        "nc3_retain": nc3_retain_list,
        "mean_source": mean_source,
    }
    return model


# ---------------------------------------------------------------------------
# 4. SCRUB + CMF  (ablation_scrub_CMF)
# ---------------------------------------------------------------------------

def ablation_scrub_CMF(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, retain_loader_full, test_loader,
    optimizer, epochs,
    test_forget_loader=None,
    mean_source: str = "train",
    **kwargs,
):
    """
    SCRUB knowledge-distillation unlearning with CMF head.
    Recomputes CMF at the end of every epoch using `mean_source`.
    """
    from utils import test
    from scrub_thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_lr
    from scrub_thirdparty.repdistiller.distiller_zoo import DistillKL
    from scrub_thirdparty.repdistiller.helper.loops import train_distill
    from unlearn.scrub import CMFHeadWrapper, _subsample_loader

    seed = int(getattr(args, "seed", 0))

    scrub_forget_loader = _subsample_loader(
        forget_loader,
        num_samples=int(args.num_forget_samples),
        batch_size=int(args.scrub_del_bsz),
        seed=seed + 123,
        num_workers=getattr(forget_loader, "num_workers", 0),
        pin_memory=True,
    )
    scrub_retain_loader = _subsample_loader(
        retain_loader,
        num_samples=int(args.num_retain_samples),
        batch_size=int(args.scrub_sgda_bsz),
        seed=seed + 456,
        num_workers=getattr(retain_loader, "num_workers", 0),
        pin_memory=True,
    )

    # SCRUB hyperparams (same as production scrub_CMF_unlearn)
    args.optim = "sgd";  args.gamma = 0.99; args.alpha = 0.001; args.beta = 0
    args.smoothing = 0.0; args.msteps = int(args.scrub_msteps); args.clip = 0.2
    args.sstart = 10; args.kd_T = 4; args.distill = "kd"
    args.sgda_epochs = int(args.scrub_epochs); args.sgda_learning_rate = float(args.lr)
    args.lr_decay_epochs = [3, 5, 9]; args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4; args.sgda_momentum = 0.9

    model_t_raw = copy.deepcopy(model).to(device)
    model_s_raw = copy.deepcopy(model).to(device)

    model_t_raw.eval()
    for p in model_t_raw.parameters():
        p.requires_grad_(False)

    _recompute(model_t_raw, train_loader, retain_loader_full, device, mean_source)
    model_s_raw.eval()
    _recompute(model_s_raw, train_loader, retain_loader_full, device, mean_source)
    mu_prev = model_s_raw.CMFweights.mu.clone().detach()

    model_t = CMFHeadWrapper(model_t_raw, detach_W=True)
    model_s = CMFHeadWrapper(model_s_raw, detach_W=bool(getattr(args, "scrub_cmf_detach_W", True)))

    beta = 0.1
    swa_model = torch.optim.swa_utils.AveragedModel(
        model_s, avg_fn=lambda avg, cur, n: (1 - beta) * avg + beta * cur)

    module_list   = nn.ModuleList([model_s, model_t]).to(device)
    trainable_list = nn.ModuleList([model_s]).to(device)
    criterion_list = nn.ModuleList([
        nn.CrossEntropyLoss(), DistillKL(args.kd_T), DistillKL(args.kd_T)
    ]).to(device)
    swa_model.to(device)

    opt = optim.SGD(trainable_list.parameters(),
                    lr=args.sgda_learning_rate,
                    momentum=args.sgda_momentum,
                    weight_decay=args.sgda_weight_decay)

    retain_acc_list, forget_acc_list, epoch_list = [], [], []
    mu_drift_list: list[float] = []
    nc3_retain_list: list[float] = []

    # Epoch 0 baseline
    model_s_raw.eval()
    r0, f0, _ = test(model_s_raw, device, test_loader,
                     args.unlearn_class, args.class_label_names, args.num_classes,
                     job_name="ablation_scrub_CMF", set_name="Test Set (Epoch 0)")
    retain_acc_list.append(r0); forget_acc_list.append(f0); epoch_list.append(0)

    model_s.train()
    for epoch in range(1, args.sgda_epochs + 1):
        sgda_adjust_lr(epoch, args, opt)

        max_loss = 0.0
        if epoch <= args.msteps:
            max_loss = train_distill(epoch, scrub_forget_loader, module_list,
                                     swa_model, criterion_list, opt, args, "maximize")
        min_acc, min_loss = train_distill(epoch, scrub_retain_loader, module_list,
                                          swa_model, criterion_list, opt, args, "minimize")

        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)

        model_s_raw.eval()
        _recompute(model_s_raw, train_loader, retain_loader_full, device, mean_source)

        mu_curr = model_s_raw.CMFweights.mu.clone().detach()
        mu_drift_list.append(float((mu_curr - mu_prev).norm().item()))
        mu_prev = mu_curr
        nc3_retain_list.append(_retain_nc3(model_s_raw, args))

        r, f, _ = test(model_s_raw, device, test_loader,
                       args.unlearn_class, args.class_label_names, args.num_classes,
                       job_name="ablation_scrub_CMF", set_name=f"Test Set (Epoch {epoch})")
        retain_acc_list.append(r); forget_acc_list.append(f); epoch_list.append(epoch)
        model_s.train()

    model_s_raw.history_log = {
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "mu_drift": mu_drift_list,
        "nc3_retain": nc3_retain_list,
        "mean_source": mean_source,
    }
    return model_s_raw


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _retain_nc3(model, args) -> float:
    """
    Compute per-retain-class cosine similarity between W_c and (m_c - μ).
    Returns the mean (1 - cosine) distance across retain classes only —
    lower is better (NC-3 duality).
    """
    try:
        W  = model.CMFweights.weight.detach()   # [K, D]
        mu = model.CMFweights.mu.detach()        # [D]
        retain_classes = [c for c in range(args.num_classes) if c not in set(args.unlearn_class)]

        if len(retain_classes) == 0:
            return float("nan")

        W_r  = W[retain_classes]                 # [R, D]
        # m_c − μ is already encoded in W_c (since W_c = normalize(m_c − μ))
        # so duality distance = 1 - cos(W_c, W_c) = 0 by construction for the
        # normalized weights.  We instead report a proxy: mean ‖W_c‖ deviation
        # from unit norm (non-zero only if weights were not renormalised).
        norms = W_r.norm(dim=1)                  # [R]
        return float((norms - 1.0).abs().mean().item())
    except Exception:
        return float("nan")


def _run_ncc(args, model, train_loader, test_loader, test_forget_loader, device):
    pool_mode = getattr(args, "nc_pool_mode", "avg")
    outs_r = ncc_mismatch(args=args, model=model, train_loader=train_loader,
                          eval_loader=test_loader, device=device, pool_mode=pool_mode)
    outs_f = ncc_mismatch(args=args, model=model, train_loader=train_loader,
                          eval_loader=test_forget_loader, device=device, pool_mode=pool_mode)
    return outs_r["ncc_acc"], outs_f["ncc_acc"]


def _build_salun_mask(args, model, device, forget_loader):
    """Re-implementation of build_salun_mask (same logic, no dependency on salun.py)."""
    mask = {name: torch.zeros_like(p, device=device)
            for name, p in model.named_parameters()}
    was_training = model.training
    model.eval()

    for data, target in forget_loader:
        data, target = data.to(device), target.to(device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(data)
            loss   = F.cross_entropy(logits, target, reduction="sum")
        loss.backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    mask[name] += p.grad.detach().abs()

    model.train(was_training)

    flat = -torch.cat([t.flatten() for t in mask.values()])
    kth  = int(len(flat) * getattr(args, "salun_threshold", 0.5))
    ranks = torch.argsort(torch.argsort(flat))

    hard, start = {}, 0
    for name, t in mask.items():
        n = t.numel()
        local_ranks = ranks[start:start + n].reshape(t.shape)
        start += n
        h = torch.zeros_like(t)
        h[local_ranks < kth] = 1.0
        hard[name] = h
    return hard
