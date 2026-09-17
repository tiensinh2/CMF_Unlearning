import torch
from unlearn.cmf_weights import CMFWeights, ModelModule


def CMF_fine_tuing(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs, **kwargs
):
    """CMF static unlearning (Algorithm 2) — NB4a entry point.

    Delegates to run_cmf_static(), which is the SINGLE SOURCE OF TRUTH for
    per-method Stage-1 logic.  run_cmf_static() dispatches to the correct
    *_CMF_unlearn() function for each base_method (grad_ascent_descent,
    random_label, salun, scrub, tarun), running exactly `epochs` epochs of:
        recompute_cmf → freeze W → method-specific encoder update.

    Previously this function ran retain-CE-only training (ignoring
    base_method's ascent/forget signal).  That caused cmf_adaptive's Stage 1
    to restore NCC_f to ~94% for grad_ascent_descent and TARUN instead of
    erasing it.  The fix is here: CMF_fine_tuing now calls run_cmf_static,
    which always uses the correct per-method logic.

    NB4a (cmf_static notebook) results are unchanged because the per-method
    *_CMF_unlearn() functions already produced the NB4a validated results.
    """
    from unlearn.cmf_two_stage import run_cmf_static

    # Derive base_method from args.unlearn_method (e.g. "scrub_CMF_RemoveFC" -> "scrub")
    unlearn_method = getattr(args, "unlearn_method", "")
    known_methods = (
        "grad_ascent_descent", "random_label", "salun", "scrub", "tarun"
    )
    base_method = next(
        (m for m in known_methods if unlearn_method.startswith(m)),
        None
    )
    if base_method is None:
        raise ValueError(
            f"CMF_fine_tuing: cannot derive base_method from "
            f"args.unlearn_method='{unlearn_method}'. "
            f"Expected one of: {known_methods}"
        )

    test_forget_loader = kwargs.pop("test_forget_loader", None)

    return run_cmf_static(
        base_method=base_method,
        args=args,
        model=model,
        device=device,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        test_forget_loader=test_forget_loader,
        **kwargs,
    )
