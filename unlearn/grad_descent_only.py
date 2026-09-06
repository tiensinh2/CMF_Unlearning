"""
unlearn/grad_descent_only.py

Gradient-descent-only unlearning (retain-set fine-tune).
This is a *retain-only* fine-tune: no gradient ascent on the forget set.
It should NOT be confused with unlearn_naive which implements NegGrad+
(ascent on forget + descent on retain).

Bug history: the dispatch table previously mapped "grad_descent" →
unlearn_naive, which runs NegGrad+ (ascent + descent). Any prior results
obtained with --unlearn-method grad_descent are affected and must be rerun.
"""
import time
import torch
import torch.nn.functional as F
from unlearn.tools import apply_prep


@apply_prep
def unlearn_grad_descent_only(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    test_forget_loader, **kwargs
):
    """Retain-only fine-tune: SGD descent on retain set, no ascent on forget.

    This is the correct implementation of gradient-descent-only unlearning.
    The method is intentionally simple: it only reinforces the retain-set
    mapping; forgetting relies on catastrophic interference.
    """
    from utils import test

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    clip = getattr(args, "grad_norm_clip", None)

    # Sub-sample retain set to args.num_retain_samples
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset,
        [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples],
    )
    gd_retain_loader = torch.utils.data.DataLoader(
        retain_dataset, batch_size=args.batch_size, shuffle=True
    )

    retain_acc_list, forget_acc_list = [], []
    epoch_list = list(range(1, epochs + 1))

    model.eval()
    test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name="grad_descent_only", set_name="Before (Test)")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        n_batches = 0
        for data, target in gd_retain_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            n_batches += 1
            if args.dry_run:
                break

        elapsed = time.time() - epoch_start
        print(f"[grad_descent_only Epoch {epoch}] {n_batches} batches, "
              f"wall={elapsed:.1f}s")

        model.eval()
        retain_acc, forget_acc, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name="grad_descent_only", set_name=f"Test Set (Epoch {epoch})",
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "epoch": epoch_list,
    }
    return model
