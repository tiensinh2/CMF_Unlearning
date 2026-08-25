"""
experiments/cmf_loader_ablation/ablation_model.py

Drop-in replacement for unlearn/cmf_weights.py that adds a `mean_source` parameter
to `recompute_cmf()`.  Everything else is identical to the production ModelModule so
that the only variable being tested is the global-mean μ computation source.

mean_source="train"  → legacy behaviour: μ computed over loader_full (forget+retain).
mean_source="retain" → ablation:         μ computed over loader_retain (retain only).

Per-class means m_c for ALL classes (including forget classes) are always computed
from their own samples so that W_c for forget classes still exists and is meaningful.

NOTE: The L2-normalize-before-averaging step is preserved exactly as in production
(z_i = normalize(f_i) before accumulation). This is a known deviation from the
original CMF paper's Algorithm 1, which averages raw features — see the original
paper for reference. That difference is out of scope for this ablation.
"""

from __future__ import annotations

import sys
import os
# allow importing from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from unlearn.cmf_weights import ModelModule as _BaseModelModule   # noqa: F401 (re-export)
from unlearn.cmf_weights import CMFWeights                         # noqa: F401 (re-export)


class AblationModelModule(_BaseModelModule):
    """
    Extends ModelModule with a two-loader recompute_cmf() that supports
    mean_source="train" (legacy) and mean_source="retain" (ablation).

    API change:
        recompute_cmf(loader_full, loader_retain, device, mean_source="train")

    Backward-compat shim (so existing single-loader call sites keep working):
        If called as recompute_cmf(single_loader, device=...) the legacy path is
        taken automatically (loader_full == loader_retain).
    """

    @torch.no_grad()
    def recompute_cmf(self, loader_full, loader_retain=None, device=None, mean_source="train"):  # type: ignore[override]
        """
        Recompute CMF classifier head.

        Args:
            loader_full:   DataLoader over the FULL training set (forget + retain).
                           Used for per-class means m_c for all classes.
            loader_retain: DataLoader over the RETAIN set only (no forget classes).
                           Used to compute global mean μ when mean_source="retain".
                           If None, falls back to loader_full regardless of mean_source.
            device:        Computation device.
            mean_source:   "train"  → μ from loader_full  (legacy, contaminated by forget)
                           "retain" → μ from loader_retain (fixed, retain-only)

        Steps (same math as production, only μ source changes):
            1. Iterate loader_full → sample-level L2 normalize → accumulate per-class sums/counts.
            2. Compute per-class mean m_c = sum_c / count_c.
            3. Compute global mean μ:
               - mean_source="train":  μ over all samples in loader_full  (includes forget)
               - mean_source="retain": μ over all samples in loader_retain (exclude forget)
            4. W_c = normalize(m_c - μ)  for every class that has samples.
        """
        # Handle legacy single-argument call: recompute_cmf(loader, device=dev)
        if device is None and not isinstance(loader_full, torch.utils.data.DataLoader):
            raise ValueError("device must be provided")
        if loader_retain is None:
            loader_retain = loader_full

        self.eval()
        K, D = self.CMFweights.num_classes, self.CMFweights.feature_dim

        # ------------------------------------------------------------------
        # Step 1 & 2: per-class sums/counts over the FULL training set
        # ------------------------------------------------------------------
        sums_full   = torch.zeros(K, D, device=device, dtype=torch.float32)
        counts_full = torch.zeros(K,    device=device, dtype=torch.float32)

        for xb, yb in loader_full:
            xb, yb = xb.to(device), yb.to(device)
            f = self.extract_features(xb).float()
            z = F.normalize(f, dim=1)       # sample-level L2 (production behaviour)
            sums_full.index_add_(0, yb, z)
            counts_full.index_add_(0, yb, torch.ones_like(yb, dtype=torch.float32))

        mask_full = counts_full > 0
        means = torch.zeros(K, D, device=device, dtype=torch.float32)
        means[mask_full] = sums_full[mask_full] / counts_full[mask_full].unsqueeze(1)

        # ------------------------------------------------------------------
        # Step 3: global mean μ — source depends on mean_source flag
        # ------------------------------------------------------------------
        if mean_source == "retain":
            # Compute μ exclusively from retain-set samples
            sums_retain   = torch.zeros(K, D, device=device, dtype=torch.float32)
            counts_retain = torch.zeros(K,    device=device, dtype=torch.float32)

            for xb, yb in loader_retain:
                xb, yb = xb.to(device), yb.to(device)
                f = self.extract_features(xb).float()
                z = F.normalize(f, dim=1)
                sums_retain.index_add_(0, yb, z)
                counts_retain.index_add_(0, yb, torch.ones_like(yb, dtype=torch.float32))

            mask_retain = counts_retain > 0
            if mask_retain.any():
                # μ = (Σ_c Σ_{i ∈ D_c^retain} ẑ_i) / N_retain_total
                mu = (sums_retain[mask_retain].sum(dim=0) /
                      counts_retain[mask_retain].sum()).detach()
            else:
                mu = torch.zeros(D, device=device)

        else:  # mean_source == "train" — legacy behaviour
            if mask_full.any():
                mu = (sums_full[mask_full].sum(dim=0) /
                      counts_full[mask_full].sum()).detach()
            else:
                mu = torch.zeros(D, device=device)

        # ------------------------------------------------------------------
        # Step 4: center & re-normalize
        # ------------------------------------------------------------------
        if mask_full.any():
            means[mask_full] = means[mask_full] - mu
            means[mask_full] = F.normalize(means[mask_full], dim=1)

        # Write back
        self.CMFweights.weight.copy_(means)
        self.CMFweights.mu.copy_(mu)
