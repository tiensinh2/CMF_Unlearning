"""
unlearn/cmf_two_stage.py — Two-stage CMF unlearning schedule.

Stage 1 (epochs 1 → E-k):
    Delegates entirely to cmf_static_unlearn (called as a black box).
    CMF weights are recomputed closed-form after each epoch.
    Encoder θ is updated by the base_method (ascent on forget + descent on retain).

Stage 2 (epochs E-k+1 → E):
    Encoder is frozen.
    W is promoted from buffer → trainable parameter ONCE (using Stage 1 end weights).
    W is trained with real gradient descent (CE loss) over retain_loader
    (or retain+forget if phase2_data='retain_plus_forget').
    A separate checkpoint is saved at Stage 1 end and Stage 2 end.

Config params:
    total_epochs      (E)  default 50
    final_stage_epochs (k) default 3   (try k=2 and k=3)
    phase2_data             'retain_only' | 'retain_plus_forget'
    mean_source             'train' | 'retain'
    base_method             method key passed to cmf_static style training

Bug fixes vs old alternating-rounds design:
    1. Old code ran ONE batch per step then broke.  This runs full epoch passes.
       We log n_batches_processed per epoch to confirm.
    2. Old code deleted buffer + re-registered parameter, breaking state_dict.
       CMFWeightsTrainable handles this cleanly; checkpoints are loadable
       in either form (buffer or parameter).
"""
from __future__ import annotations

import copy
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unlearn.tools import apply_prep


# ─────────────────────────────────────────────────────────────────────────────
# CMFWeightsTrainable
# ─────────────────────────────────────────────────────────────────────────────

class CMFWeightsTrainable(nn.Module):
    """Wraps a CMFWeights buffer and can promote it to a trainable parameter.

    Design: we never delete the original buffer.  Instead we store a separate
    nn.Parameter that shadows the buffer.  When trainable=False the forward
    returns the buffer value; when trainable=True it returns the parameter.
    This means state_dict() always contains BOTH keys, so checkpoints saved
    in either mode can be loaded in either mode with strict=False.

    Usage:
        tw = CMFWeightsTrainable(cmf_weights_module)
        # ... Stage 1 ends ...
        tw.promote()          # snapshot buffer → parameter, set trainable=True
        optim = SGD([tw.W_param], lr=lr)
        # ... Stage 2 gradient steps on tw.W_param ...
        tw.sync_back()        # write parameter value back to buffer (optional)
    """

    def __init__(self, cmf_weights_module: nn.Module):
        super().__init__()
        self._cmf = cmf_weights_module
        # Shadow parameter — initialised to zeros, promoted on demand
        w0 = cmf_weights_module.weight.detach().clone()
        # Register as a buffer initially so it doesn't appear in parameters()
        self.register_buffer("_W_buf_shadow", w0)
        self._W_param: Optional[nn.Parameter] = None
        self.trainable: bool = False

    @property
    def W_param(self) -> nn.Parameter:
        if self._W_param is None:
            raise RuntimeError("Call .promote() before accessing W_param.")
        return self._W_param

    def promote(self) -> None:
        """Snapshot the current buffer value and register as nn.Parameter."""
        if self.trainable:
            return  # already promoted
        w_val = self._cmf.weight.detach().clone()
        self._W_param = nn.Parameter(w_val)
        # Register it so optimizer can find it
        self.register_parameter("W_param_inner", self._W_param)
        self.trainable = True

    def forward(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute logits = z @ W.T * temperature."""
        W = self._W_param if self.trainable else self._cmf.weight
        return (z @ W.t()) * temperature

    def sync_back(self) -> None:
        """Write the trained parameter value back into the CMFWeights buffer."""
        if self._W_param is not None:
            with torch.no_grad():
                self._cmf.weight.copy_(self._W_param.data)

    def state_dict_extra(self) -> dict:
        """Extra state for checkpointing that captures both forms."""
        d = {"trainable": self.trainable}
        if self._W_param is not None:
            d["W_param"] = self._W_param.data.clone()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _save_ckpt(model, path: str, extra: dict = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[ckpt] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage-2: train W with gradient descent
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage2(
    model: nn.Module,
    tw: CMFWeightsTrainable,
    retain_loader,
    forget_loader,
    device: torch.device,
    k_epochs: int,
    lr: float,
    phase2_data: str,
    temperature: float,
    args,
    epoch_offset: int,
    log_rows: list,
    test_fn,
    test_loader,
    lp_fn,
    train_loader,
) -> None:
    """Run Stage 2: freeze encoder, gradient-descend W."""
    import evaluation
    from utils import test

    # Freeze encoder
    for p in model.parameters():
        p.requires_grad_(False)
    tw.promote()

    optim_w = torch.optim.SGD([tw.W_param], lr=lr, momentum=0.9, weight_decay=1e-4)

    # Build Stage-2 loader
    if phase2_data == "retain_plus_forget" and forget_loader is not None:
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([retain_loader.dataset, forget_loader.dataset])
        s2_loader = torch.utils.data.DataLoader(
            combined, batch_size=retain_loader.batch_size, shuffle=True
        )
    else:
        s2_loader = retain_loader

    t_start = time.time()
    for ep in range(1, k_epochs + 1):
        model.train()  # needed for BN running stats even if weights frozen
        n_batches = 0
        ep_loss = 0.0
        for xb, yb in s2_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim_w.zero_grad()
            with torch.no_grad():
                f = model.extract_features(xb)
                z = model._preprocess_feats_for_cmf(f)
            logits = tw(z, temperature)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim_w.step()
            ep_loss += loss.item()
            n_batches += 1
            if getattr(args, "dry_run", False):
                break

        # Sync trained W back into buffer so test() / recompute_cmf() see it
        tw.sync_back()

        model.eval()
        retain_acc, forget_acc, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name="cmf_two_stage_s2", set_name=f"S2 Epoch {ep}",
        )

        # Linear probe
        lp_ret, lp_for = None, None
        bs_probe = getattr(args, "prob_batch_size", 256)
        if lp_fn is not None and getattr(args, "lp_every", 1) > 0:
            try:
                out_lp = lp_fn(
                    args=args, get_model_fn=None, device_probe=device,
                    src_model=model, train_loader=train_loader,
                    test_loader=test_loader, num_classes=args.num_classes,
                    bs_probe=bs_probe,
                )
                lp_ret = out_lp.get("acc_test_retain")
                lp_for = out_lp.get("acc_test_forget")
            except Exception as e:
                print(f"[Stage2 LP] skipped: {e}")

        # NCC
        ncc_ret, ncc_for = None, None
        try:
            from evaluation.nc import ncc_mismatch
            ret_res = ncc_mismatch(args, model, retain_loader, retain_loader, device)
            ncc_ret = ret_res["ncc_acc"]
            for_res = ncc_mismatch(args, model, retain_loader,
                                    torch.utils.data.DataLoader(
                                        forget_loader.dataset, batch_size=256),
                                    device)
            ncc_for = for_res["ncc_acc"]
        except Exception as e:
            print(f"[Stage2 NCC] skipped: {e}")

        wall = (time.time() - t_start) / 60.0
        log_rows.append({
            "epoch_stage": f"S2_ep{ep}",
            "output_retain_acc": retain_acc,
            "output_forget_acc": forget_acc,
            "probe_retain_acc": lp_ret,
            "probe_forget_acc": lp_for,
            "ncc_retain_acc": ncc_ret,
            "ncc_forget_acc": ncc_for,
            "wall_clock_minutes": wall,
            "n_batches": n_batches,
            "stage2_loss": ep_loss / max(n_batches, 1),
        })
        print(f"[S2 ep{ep}] retain={retain_acc:.4f} forget={forget_acc:.4f} "
              f"batches={n_batches} loss={ep_loss/max(n_batches,1):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main: cmf_two_stage_unlearn
# ─────────────────────────────────────────────────────────────────────────────

@apply_prep
def cmf_two_stage_unlearn(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    test_forget_loader, **kwargs
):
    """Two-stage CMF unlearning.

    Stage 1: cmf_static_unlearn as black box (epochs 1 → E-k).
    Stage 2: freeze encoder, gradient-descend W (epochs E-k+1 → E).

    Required args attributes:
        total_epochs        int   (E, default 50)
        final_stage_epochs  int   (k, default 3)
        phase2_data         str   'retain_only' | 'retain_plus_forget'
        mean_source         str   'train' | 'retain'
        base_method         str   method key for Stage 1
        lr                  float
        momentum, weight_decay, batch_size (standard)
    """
    from utils import test, get_model
    import evaluation

    E = getattr(args, "total_epochs", 50)
    k = getattr(args, "final_stage_epochs", 3)
    phase2_data = getattr(args, "phase2_data", "retain_only")
    mean_source = getattr(args, "mean_source", "train")
    temperature = getattr(model, "args", args).temperature if hasattr(
        getattr(model, "args", None), "temperature") else getattr(args, "temperature", 1.0)

    E = max(E, k + 1)   # ensure at least 1 stage-1 epoch
    stage1_epochs = E - k

    print(f"[cmf_two_stage] E={E} k={k} stage1={stage1_epochs} "
          f"phase2_data={phase2_data} mean_source={mean_source}")

    log_rows: list = []
    t0 = time.time()

    # ─── Baseline eval (epoch 0) ───────────────────────────────────────────
    model.eval()
    mean_loader = train_loader if mean_source == "train" else retain_loader
    model.recompute_cmf(mean_loader, device=device)
    retain_acc0, forget_acc0, _ = test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name="cmf_two_stage", set_name="Epoch 0",
    )
    log_rows.append({
        "epoch_stage": "S1_ep0",
        "output_retain_acc": retain_acc0,
        "output_forget_acc": forget_acc0,
        "probe_retain_acc": None,
        "probe_forget_acc": None,
        "ncc_retain_acc": None,
        "ncc_forget_acc": None,
        "wall_clock_minutes": 0.0,
        "n_batches": 0,
        "stage2_loss": None,
    })

    # ─── Stage 1: delegate to cmf_static ──────────────────────────────────
    # Import here to avoid circular dependency
    from unlearn.CMF import CMF_fine_tuing as cmf_static_unlearn

    # Override epochs so cmf_static runs only stage1_epochs
    orig_epochs = epochs
    # We monkey-patch args temporarily
    _save_epochs = getattr(args, "_two_stage_s1_epochs", None)
    args._two_stage_s1_epochs = stage1_epochs

    # Build a thin wrapper around cmf_static that logs per-epoch rows
    print(f"\n{'='*60}")
    print(f"[cmf_two_stage] Stage 1: running {stage1_epochs} epochs via cmf_static")
    print(f"{'='*60}\n")

    # cmf_static uses args.epochs_or_steps — temporarily override
    orig_eps = getattr(args, "epochs_or_steps", epochs)
    args.epochs_or_steps = stage1_epochs

    model = cmf_static_unlearn(
        args=args,
        model=model,
        device=device,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=stage1_epochs,
        test_forget_loader=test_forget_loader,
        **{k2: v for k2, v in kwargs.items()},
    )

    args.epochs_or_steps = orig_eps

    # Capture Stage-1 history from model.history_log
    if hasattr(model, "history_log"):
        h = model.history_log
        for ep_idx, ep_num in enumerate(h.get("epoch", [])):
            if ep_num == 0:
                continue  # skip epoch-0 already logged
            row = {
                "epoch_stage": f"S1_ep{ep_num}",
                "output_retain_acc": h["retain_acc"][ep_idx] if ep_idx < len(h.get("retain_acc", [])) else None,
                "output_forget_acc": h["forget_acc"][ep_idx] if ep_idx < len(h.get("forget_acc", [])) else None,
                "probe_retain_acc": h["LP_retain_acc"][ep_idx] if ep_idx < len(h.get("LP_retain_acc", [])) else None,
                "probe_forget_acc": h["LP_forget_acc"][ep_idx] if ep_idx < len(h.get("LP_forget_acc", [])) else None,
                "ncc_retain_acc": None,
                "ncc_forget_acc": None,
                "wall_clock_minutes": (time.time() - t0) / 60.0,
                "n_batches": None,
                "stage2_loss": None,
            }
            log_rows.append(row)

    # ─── Save Stage-1 checkpoint ──────────────────────────────────────────
    ckpt_dir = f"./checkpoints/cmf_two_stage/{getattr(args, 'dataset', 'cifar10')}_{getattr(args, 'arch', 'resnet18')}"
    classes_str = ",".join(str(c) for c in getattr(args, "unlearn_class", []))
    s1_ckpt = f"{ckpt_dir}/{classes_str}_stage1.pt"
    _save_ckpt(model, s1_ckpt, extra={"stage": 1, "epoch": stage1_epochs,
                                       "official_repo_commit": getattr(args, "official_repo_commit", "main")})

    # ─── Stage 2: promote W → trainable, gradient descent ────────────────
    print(f"\n{'='*60}")
    print(f"[cmf_two_stage] Stage 2: {k} epochs of W gradient descent "
          f"(phase2_data={phase2_data})")
    print(f"{'='*60}\n")

    # Ensure CMF weights buffer is current after Stage 1
    model.eval()
    model.recompute_cmf(mean_loader, device=device)

    tw = CMFWeightsTrainable(model.CMFweights)
    tw = tw.to(device)

    lp_fn = None
    try:
        lp_fn = evaluation.run_linear_probe_on_fresh_clone
    except AttributeError:
        pass

    _run_stage2(
        model=model,
        tw=tw,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        device=device,
        k_epochs=k,
        lr=getattr(args, "lr", 1e-3),
        phase2_data=phase2_data,
        temperature=temperature,
        args=args,
        epoch_offset=stage1_epochs,
        log_rows=log_rows,
        test_fn=test,
        test_loader=test_loader,
        lp_fn=lp_fn,
        train_loader=train_loader,
    )

    # ─── Save Stage-2 checkpoint ──────────────────────────────────────────
    s2_ckpt = f"{ckpt_dir}/{classes_str}_stage2_k{k}_{phase2_data}.pt"
    _save_ckpt(model, s2_ckpt, extra={
        "stage": 2, "epoch": E, "k": k, "phase2_data": phase2_data,
        "tw_extra": tw.state_dict_extra(),
        "official_repo_commit": getattr(args, "official_repo_commit", "main"),
    })

    model.history_log = {
        "two_stage_log": log_rows,
        "total_epochs": E,
        "stage1_epochs": stage1_epochs,
        "k": k,
        "phase2_data": phase2_data,
        "mean_source": mean_source,
    }
    return model
