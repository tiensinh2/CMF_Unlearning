"""
evaluation/nc.py — NC metrics for both standard and CMF models.

Aligned with the official ycgao1/CMF_Unlearning repo (evaluation/nc_cmf.py).
Key fixes vs the local fork:
  1. get_classifier_weights() — 5-tier fallback correctly returns
     model.CMFweights.weight for CMF models instead of the Identity fc.
  2. _features_for_nc() — applies the CMF geometry transform
     ẑ = normalize(normalize(f) − μ) when the model has _preprocess_feats_for_cmf,
     making NC metrics geometrically consistent with the classifier.
  3. ncc_accuracy_from_features() / ncc_mismatch() — NCC (NC-4) implemented as
     nearest-class-center in feature space, replacing the broken identity-matrix path.
  4. duality_distance() — NC-3 computed as mean (1 − cos(h_c, w_c)).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as scilin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear


# ─────────────────────────────────────────────────────────────────────────────
# Utility: pool 4-D feature maps
# ─────────────────────────────────────────────────────────────────────────────

def pool_tensor(tensor: torch.Tensor, mode: str = "avg") -> torch.Tensor:
    """Pool [B,C,H,W] → [B,C]; pass [B,C] unchanged."""
    if tensor.dim() == 4:
        if mode == "avg":
            out = F.adaptive_avg_pool2d(tensor, 1)
        else:
            out = F.adaptive_max_pool2d(tensor, 1)
        return out.view(out.size(0), -1)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 fix: correct classifier-weight extraction for CMF models
# ─────────────────────────────────────────────────────────────────────────────

def get_classifier_weights(
    model: nn.Module,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the classifier weight matrix W [C, D].

    Five-tier fallback (aligned with official nc_cmf.py):
      1. model.cmf_weights  attribute
      2. model.CMFweights.weight  buffer  ← CMF models land here
      3. model.classifier / .fc / .head / .linear  (nn.Linear)
      4. Last nn.Linear in model.modules()
      5. Any 2-D parameter/buffer with one dim == num_classes
    """
    def _shape_ok(W: torch.Tensor) -> Optional[torch.Tensor]:
        if W.ndim != 2:
            return None
        if W.shape[0] == num_classes:
            return W
        if W.shape[1] == num_classes:
            return W.t()
        return None

    # 1) explicit cmf_weights attribute
    if hasattr(model, "cmf_weights"):
        W = getattr(model, "cmf_weights").detach().to(device)
        r = _shape_ok(W)
        if r is not None:
            return r

    # 2) CMFWeights buffer (CMF models)
    if hasattr(model, "CMFweights"):
        cmf_mod = getattr(model, "CMFweights")
        if hasattr(cmf_mod, "weight"):
            W = cmf_mod.weight.detach().to(device)
            r = _shape_ok(W)
            if r is not None:
                return r

    # 3) common head attribute names
    for attr in ("classifier", "fc", "head", "linear"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            W = None
            if isinstance(head, nn.Linear):
                W = head.weight
            elif hasattr(head, "weight") and isinstance(head.weight, torch.Tensor):
                W = head.weight
            if W is not None:
                W = W.detach().to(device)
                r = _shape_ok(W)
                if r is not None:
                    return r

    # 4) last nn.Linear in the model
    last_linear: Optional[nn.Linear] = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        W = last_linear.weight.detach().to(device)
        r = _shape_ok(W)
        if r is not None:
            return r

    # 5) any 2-D parameter / buffer with correct shape
    for _name, p in model.named_parameters():
        W = p.detach().to(device)
        r = _shape_ok(W)
        if r is not None:
            return r
    for _name, b in model.named_buffers():
        W = b.detach().to(device)
        r = _shape_ok(W)
        if r is not None:
            return r

    raise RuntimeError(
        f"No suitable classifier weight found in model for num_classes={num_classes}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier-2 fix: CMF-aware feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _features_for_nc(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    use_cmf: bool,
    pool_mode: str = "avg",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features [N, D] and labels [N] for NC metric computation.

    For CMF models (use_cmf=True) we apply the full geometry transform
        ẑ = normalize(normalize(f) − μ)
    via model._preprocess_feats_for_cmf(), keeping the feature space
    consistent with the prototype matrix W.

    For standard models we extract head-input features without normalisation.
    """
    model.eval().to(device)
    Xs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        if use_cmf:
            f = model.extract_features(xb)
            z = model._preprocess_feats_for_cmf(f)
        else:
            # Hook into the penultimate layer (one before the final Linear)
            modules = [
                m for _, m in model.named_modules()
                if isinstance(m, (Conv2d, Linear))
            ]
            if len(modules) < 2:
                raise RuntimeError("Model too small: need at least 2 Conv2d/Linear layers.")
            pen_mod = modules[-2]
            buf: List[torch.Tensor] = []
            h = pen_mod.register_forward_hook(
                lambda _m, _i, out: buf.append(pool_tensor(out, pool_mode).detach())
            )
            _ = model(xb)
            h.remove()
            z = buf[0]
        Xs.append(z.cpu().float())
        ys.append(yb.cpu())
    return torch.cat(Xs, 0), torch.cat(ys, 0)


# ─────────────────────────────────────────────────────────────────────────────
# NC-3: duality distance (mean 1 − cosine alignment)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mu_G_and_mu_c(
    X: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    mu_G = X.mean(dim=0)
    mu_c: Dict[int, torch.Tensor] = {}
    for c in torch.unique(y).tolist():
        c = int(c)
        mask = y == c
        if mask.any():
            mu_c[c] = X[mask].mean(dim=0)
    return mu_G, mu_c


def duality_distance(
    mu_c_dict: Dict[int, torch.Tensor],
    mu_G: torch.Tensor,
    W: torch.Tensor,
) -> Tuple[float, float, float]:
    """NC-3 as average (1 − cos(h_c, w_c)) where h_c = μ_c − μ_G."""
    if not mu_c_dict:
        return 0.0, float(torch.norm(W).item()), 0.0
    C = W.shape[0]
    device = W.device
    h = torch.zeros_like(W, device=device)
    rows: List[int] = []
    for c, mu_c in mu_c_dict.items():
        if 0 <= c < C:
            h[c] = mu_c.to(device) - mu_G.to(device)
            rows.append(c)
    if not rows:
        return float(torch.norm(h).item()), float(torch.norm(W).item()), 0.0
    h_norm = torch.norm(h).item()
    w_norm = torch.norm(W).item()
    h_n = F.normalize(h, dim=1, p=2)
    W_n = F.normalize(W, dim=1, p=2)
    dist = sum(1.0 - torch.dot(h_n[i], W_n[i]).item() for i in rows) / len(rows)
    return float(h_norm), float(w_norm), float(dist)


# ─────────────────────────────────────────────────────────────────────────────
# NC-4: NCC accuracy
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ncc_accuracy_from_features(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_ev: torch.Tensor,
    y_ev: torch.Tensor,
    num_classes: int,
) -> Tuple[float, torch.Tensor]:
    """Nearest-class-centre accuracy: train centres → classify eval set."""
    device = X_tr.device
    means = []
    for c in range(num_classes):
        mask = y_tr == c
        mu = X_tr[mask].mean(dim=0) if mask.any() else torch.zeros(X_tr.size(1), device=device)
        means.append(mu)
    M = torch.stack(means)                       # [C, D]
    dists = torch.cdist(X_ev.unsqueeze(0), M.unsqueeze(0)).squeeze(0)  # [N, C]
    pred = dists.argmin(dim=1)
    correct = (pred == y_ev.to(device)).float()
    acc_all = correct.mean().item()
    acc_per = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        m = y_ev.to(device) == c
        if m.any():
            acc_per[c] = correct[m].mean()
        else:
            acc_per[c] = float("nan")
    return float(acc_all), acc_per.cpu()


@torch.no_grad()
def ncc_mismatch(
    args,
    model: nn.Module,
    train_loader,
    eval_loader,
    device: torch.device,
    *,
    pool_mode: str = "avg",
) -> Dict[str, object]:
    """NCC accuracy on eval_loader using class centres from train_loader."""
    device = device if isinstance(device, torch.device) else torch.device(device)
    use_cmf = (
        "CMF" in getattr(args, "unlearn_method", "")
        and hasattr(model, "_preprocess_feats_for_cmf")
    )
    X_tr, y_tr = _features_for_nc(model, train_loader, device, use_cmf=use_cmf, pool_mode=pool_mode)
    X_ev, y_ev = _features_for_nc(model, eval_loader, device, use_cmf=use_cmf, pool_mode=pool_mode)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_ev, y_ev = X_ev.to(device), y_ev.to(device)
    acc_all, acc_per = ncc_accuracy_from_features(X_tr, y_tr, X_ev, y_ev, args.num_classes)
    return {
        "ncc_acc": float(acc_all),
        "ncc_mismatch": 1.0 - float(acc_all),
        "ncc_acc_per_class": acc_per.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy NC-1 / NC-2 / NC-3-traditional helpers (kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def _class_means(feats, labels, classes):
    return torch.stack([feats[labels == c].mean(0) for c in classes])


def _build_Sw_Sb(feats, labels, classes, use_global):
    K = len(classes)
    mu_c = _class_means(feats, labels, classes)
    if use_global:
        mu_G = feats.mean(0)
    else:
        mask = torch.isin(labels, torch.tensor(classes, device=feats.device))
        mu_G = feats[mask].mean(0)
    Sw = torch.zeros((feats.size(1), feats.size(1)), device=feats.device)
    for i, c in enumerate(classes):
        xi = feats[labels == c]
        diff = xi - mu_c[i]
        Sw += (diff.t() @ diff) / xi.size(0)
    Sw /= K
    diffc = (mu_c - mu_G).T
    Sb = diffc @ diffc.T / K
    return Sw, Sb, mu_c, mu_G


def _nc1(Sw, Sb, K):
    return (torch.trace(
        Sw @ torch.tensor(scilin.pinv(Sb.cpu().numpy()), device=Sw.device)
    ) / K).item()


def _nc2(mu_c, mu_G):
    K = mu_c.size(0)
    H = (mu_c - mu_G).T
    HH = H.T @ H
    HH /= torch.norm(HH, p="fro")
    ETF = (torch.eye(K, device=mu_c.device) - torch.ones(K, device=mu_c.device) / K) / ((K - 1) ** 0.5)
    return torch.norm(HH - ETF, p="fro").item()


def _nc3_traditional(W, mu_c, mu_G):
    K = mu_c.size(0)
    H = (mu_c.to(W.device) - mu_G.to(W.device)).T
    WH = W @ H
    WHn = WH / torch.norm(WH, p="fro")
    ETF = (torch.eye(K, device=W.device) - torch.ones(K, device=W.device) / K) / ((K - 1) ** 0.5)
    return torch.norm(WHn - ETF, p="fro").item()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point: nc_metrics()
# ─────────────────────────────────────────────────────────────────────────────

def nc_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    retain_classes: List[int],
    forget_classes: List[int],
    *,
    train_loader=None,
    args=None,
    pool_mode: str = "avg",
) -> Dict:
    """Compute NC-1, NC-2, NC-3 (duality), NCC (NC-4) on *loader*.

    Uses get_classifier_weights() so CMF models get the correct W.
    Uses _features_for_nc() so CMF models get the correct ẑ geometry.

    Args:
        loader:       Eval loader (test set).  Features extracted from here
                      are used for NC-3 and as the *evaluation* side of NCC.
        train_loader: Optional.  When provided, class-mean centres μ_k for NCC
                      are computed from this loader (matching the paper protocol:
                      centres from train, accuracy evaluated on test).
                      When None, centres fall back to *loader* (self-evaluation).
    """
    device = device if isinstance(device, torch.device) else torch.device(device)
    all_classes = sorted(set(retain_classes) | set(forget_classes))
    num_classes = max(all_classes) + 1 if all_classes else 10

    # Determine CMF mode
    use_cmf = (
        args is not None
        and "CMF" in getattr(args, "unlearn_method", "")
        and hasattr(model, "_preprocess_feats_for_cmf")
    )

    # Correct classifier weights
    W = get_classifier_weights(model, num_classes, device)  # [C, D]

    # Eval-set features (used for NC-3 and as NCC evaluation split)
    X, y = _features_for_nc(model, loader, device, use_cmf=use_cmf, pool_mode=pool_mode)
    X, y = X.to(device), y.to(device)

    # --- NC-3 duality distance (computed on eval-set features) ---
    mu_G, mu_c = _compute_mu_G_and_mu_c(X, y)
    h_norm, w_norm, nc3_dist = duality_distance(mu_c, mu_G, W)

    # --- NCC (NC-4) ---
    # Paper protocol: centres from training data, accuracy on eval (test) data.
    # When train_loader is supplied we compute train-set centres; otherwise we
    # fall back to using eval-set centres (self-evaluation).
    if train_loader is not None:
        X_tr, y_tr = _features_for_nc(
            model, train_loader, device, use_cmf=use_cmf, pool_mode=pool_mode
        )
        X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    else:
        X_tr, y_tr = X, y  # fallback: self-evaluation (original behaviour)
    ncc_all, ncc_per_class = ncc_accuracy_from_features(X_tr, y_tr, X, y, num_classes)

    # --- NC-1 / NC-2 (legacy, on all classes) ---
    try:
        Sw_all, Sb_all, mu_all, muG_all = _build_Sw_Sb(X, y, all_classes, False)
        nc1_all = _nc1(Sw_all, Sb_all, len(all_classes))
        nc2_all = _nc2(mu_all, muG_all)
        Sw_r, Sb_r, mu_r, muG_r = _build_Sw_Sb(X, y, retain_classes, True)
        nc1_ret = _nc1(Sw_r, Sb_r, len(retain_classes))
        nc2_ret = _nc2(mu_r, muG_r)
        nc1_for = None
        if forget_classes:
            Sw_f, Sb_f, mu_f, muG_f = _build_Sw_Sb(X, y, forget_classes, True)
            nc1_for = _nc1(Sw_f, Sb_f, len(forget_classes))
    except Exception:
        nc1_all = nc2_all = nc1_ret = nc2_ret = nc1_for = None

    # NCC split by retain/forget
    retain_mask = torch.isin(y, torch.tensor(retain_classes, device=device))
    forget_mask = torch.isin(y, torch.tensor(forget_classes, device=device))
    # Use all-data centres, evaluate on retain/forget subsets
    ncc_ret = ncc_per_class[retain_classes].nanmean().item() if retain_classes else None
    ncc_for = ncc_per_class[forget_classes].nanmean().item() if forget_classes else None

    return dict(
        NC1_all=nc1_all,   NC1_ret=nc1_ret,   NC1_for=nc1_for,
        NC2_all=nc2_all,   NC2_ret=nc2_ret,
        NC3_duality=nc3_dist,
        NC3_h_norm=h_norm, NC3_w_norm=w_norm,
        NCC_all=ncc_all,
        NCC_retain=ncc_ret,
        NCC_forget=ncc_for,
        ncc_per_class=ncc_per_class.tolist(),
    )
