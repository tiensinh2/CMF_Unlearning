import torch
import torch.nn.functional as F
from torch import nn

from unlearn.cmf_weights import CMFWeights, ModelModule

import os

def CMF_fine_tuing(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs, **kwargs
):
    """CMF static unlearning (Algorithm 2).

    train_loader is the mean_loader passed by NB4/NB5 — either the full
    training set ('train') or retain-only ('retain') depending on mean_source.
    recompute_cmf and the per-epoch mean update both use this same loader so
    that mean_source='retain' correctly computes means from retain data only.

    Bug C1 fix: recompute_cmf now uses train_loader (= mean_loader passed by
    caller), not a hardcoded reference to the full training loader.
    Bug C2 fix: the gradient-update loop now trains on retain_loader only,
    never on forget-class samples.
    """
    from utils import test

    # Align CMF weights using mean_loader (passed as train_loader by NB4/NB5)
    model.eval()
    model.recompute_cmf(train_loader, device=device)

    # SGD optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=getattr(args, "lr", 1e-3),
        momentum=getattr(args, "momentum", 0.9),
        weight_decay=getattr(args, "weight_decay", 5e-4),
        nesterov=True
    )

    # Training loop — retain_loader only (never trains on forget class)
    model.train()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss, acc = model.forward_a((x, y), stage="train")
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_loss   += loss.item() * bs
            total_correct += float(acc.item()) * bs
            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc  = total_correct / max(total_samples, 1)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}")

        # Recompute CMF using mean_loader after each epoch
        model.eval()
        model.recompute_cmf(train_loader, device=device)
        retain_acc, forget_acc, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
        )
        model.train()

    # Final recompute and evaluation
    model.eval()
    model.recompute_cmf(train_loader, device=device)
    retain_acc, forget_acc, _ = test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (Final)"
    )
    model.history_log = {
        "retain_acc": [retain_acc],
        "forget_acc": [forget_acc],
    }
    return model
