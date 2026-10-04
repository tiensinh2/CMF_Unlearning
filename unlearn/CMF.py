import torch
import torch.nn.functional as F
from torch import nn

from unlearn.cmf_weights import CMFWeights, ModelModule

import os


# ---------------------------------------------------------------------------
# run_cmf_static
# ---------------------------------------------------------------------------
# Implements paper Algorithm 2 (Appendix B.1): gradient-based unlearning with
# CMF head reconstruction.  Dispatches to the appropriate *_CMF_RemoveFC
# function for the chosen base_method.
#
# Signature matches the call sites in notebooks/cifar10/04_cmf_static.ipynb
# and notebooks/cifar100/04_cmf_static.ipynb:
#
#   model = run_cmf_static(
#       base_method=base_method,
#       args=args, model=model, device=device,
#       retain_loader=retain_loader, forget_loader=forget_loader,
#       train_loader=mean_loader,   test_loader=test_loader_full,
#       epochs=cmf_epochs,          test_forget_loader=test_forget_ldr,
#       train_dataset=full_train,   val_index=retain_train_idx,
#   )
# ---------------------------------------------------------------------------

_CMF_DISPATCH = {
    "grad_ascent_descent": "grad_ascent_descent_CMF_RemoveFC",
    "random_label":        "random_label_CMF_RemoveFC",
    "salun":               "salun_CMF_RemoveFC",
    "scrub":               "scrub_CMF_RemoveFC",
    "tarun":               "tarun_CMF_RemoveFC",
}


def run_cmf_static(
    base_method,
    args,
    model,
    device,
    retain_loader,
    forget_loader,
    train_loader,
    test_loader,
    epochs,
    test_forget_loader=None,
    train_dataset=None,
    val_index=None,
    **kwargs,
):
    """
    Paper Algorithm 2 dispatcher.

    Looks up the CMF-augmented variant of *base_method* in unlear_func and
    calls it with the standard unlearn function signature.  The caller is
    responsible for building/loading the ModelModule and pre-computing the
    initial CMF head before calling this function.

    Parameters
    ----------
    base_method : str
        One of 'grad_ascent_descent', 'random_label', 'salun', 'scrub',
        'tarun'.
    args        : argparse.Namespace  — hyperparameters (lr, epochs, …)
    model       : ModelModule         — CMF model (encoder + CMFweights head)
    device      : torch.device
    retain_loader / forget_loader / train_loader / test_loader : DataLoader
    epochs      : int
    test_forget_loader : DataLoader, optional
    train_dataset      : Dataset, optional  (required by tarun / SVD)
    val_index          : list,    optional  (required by tarun / SVD)
    """
    from unlearn import unlear_func  # imported here to avoid circular imports

    dispatch_key = _CMF_DISPATCH.get(base_method)
    if dispatch_key is None:
        raise ValueError(
            f"run_cmf_static: unknown base_method '{base_method}'. "
            f"Supported: {sorted(_CMF_DISPATCH)}"
        )
    fn = unlear_func.get(dispatch_key)
    if fn is None:
        raise KeyError(
            f"run_cmf_static: dispatch key '{dispatch_key}' not found in "
            f"unlear_func.  Available keys: {sorted(unlear_func)}"
        )

    return fn(
        args=args,
        model=model,
        device=device,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=None,
        epochs=epochs,
        test_forget_loader=test_forget_loader,
        train_dataset=train_dataset,
        val_index=val_index,
        **kwargs,
    )

def CMF_fine_tuing(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs, **kwargs
):
    from utils import test
    model.eval()
    model.recompute_cmf(train_loader, device=device)   # ★ key：andlogic/trainingconsistent's/of W、μ
    model.train()

    '''''
    # Step 0: Test old model before fine-tuning
    print("[Before Fine-tuning] Testing old model")
    test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (Before Fine-tune)"
    )

    # Step 1: constructuse CMF 's/ofnewmodel
    args.CMFClassifier = True  # ✅ use CMF classifier
    new_model = ModelModule(args).to(device)

    # Step 2: control encoder number（modelis encoder）
    new_model.encoder.load_state_dict(model.state_dict())
    '''''


    # Step 3:  SGD optimalifydevice
    optimizer = torch.optim.SGD(
        #filter(lambda p: p.requires_grad, new_model.parameters()),
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=getattr(args, "lr", 1e-3),
        momentum=getattr(args, "momentum", 0.9),
        weight_decay=getattr(args, "weight_decay", 5e-4),
        nesterov=True
    )

    # Step 4: Fine-tuning Loop
    #new_model.train()
    model.train()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch = (x, y)

            optimizer.zero_grad()

            #loss, acc = new_model.forward_a(batch, stage="train")
            loss, acc = model.forward_a(batch, stage="train")

            
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_correct += float(acc.item()) * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}")

        # Step 5: Eval on Test Set
        #new_model.eval()
        model.eval()
        model.recompute_cmf(train_loader, device=device)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                #feats = F.normalize(new_model.encoder(x))
                #weights = F.normalize(new_model.CMFweights.weight)
                logits = model.forward(x)   # ★ sampleuseuse ẑ @ W^T
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / max(1, total)
        print(f"[Epoch {epoch}] CMF Test Accuracy: {acc:.2f}%")
        #new_model.train()

        test(
        #new_model, device, test_loader,
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (After Fine-tune)"
        )
        

    # Step 6: Final Evaluation
    print("[After Fine-tuning] Testing CMF model")
    model.recompute_cmf(train_loader, device=device)
    test(
        #new_model, device, test_loader,
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (After Fine-tune)"
    )

    #return new_model
    return model

