# evaluation/__init__.py
# Restored: all public symbols needed by main.py, unlearn/naive.py, and notebooks.
# nc_metrics now uses get_classifier_weights() + _features_for_nc() (CMF-aware).
from .nc import (
    nc_metrics,
    get_classifier_weights,
    _features_for_nc,
    ncc_accuracy_from_features,
    ncc_mismatch,
    duality_distance,
)
from .linear_prob import (
    linear_probe_CMF_RemoveFC,
    run_linear_probe_on_fresh_clone,
    linear_probe_last_layer,
)
