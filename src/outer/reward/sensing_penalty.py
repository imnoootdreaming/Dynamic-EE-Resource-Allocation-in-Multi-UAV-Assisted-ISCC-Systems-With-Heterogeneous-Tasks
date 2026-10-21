"""Bounded sensing-shortfall score for the existing no-solution penalty.

The margin is the best achievable sensing SINR divided by its required
threshold, for the current geometry, receive beam, CU matching and powers.
This module scores that necessary condition; it does not certify the complete
inner problem and does not change any channels or actions.
"""

import numpy as np


SENSING_PENALTY_CONTRACT = "normalized_sensing_shortfall_v1"


def validate_sensing_penalty_settings(coefficient, reference_factor):
    """Return finite float settings with coefficient in [0, 1] and F > 1."""
    values = []
    for name, value in (("sensing_penalty_coef", coefficient),
                        ("sensing_penalty_reference_factor", reference_factor)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a number, not a boolean")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        values.append(value)
    coefficient, reference_factor = values
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("sensing_penalty_coef must be between 0 and 1")
    if reference_factor <= 1.0:
        raise ValueError("sensing_penalty_reference_factor must be greater than 1")
    return coefficient, reference_factor


def normalized_sensing_shortfall(margins, reference_factor=10.0, *,
                                expected_uavs=None):
    """Return (mean score, per-UAV scores) without modifying the margins.

    For 0 < M < 1, d = -log(M) and q = d / (d + log(F)). A satisfied
    sensing condition M >= 1 has q = 0; M = 0 has the limiting score q = 1.
    Thus M = 1/F scores 0.5, and the mean stays in [0, 1] for any UAV count.
    Invalid diagnostic values raise ValueError for the caller to handle.
    """
    _, reference_factor = validate_sensing_penalty_settings(0.0, reference_factor)
    if np.iscomplexobj(margins):
        raise ValueError("sensing margins must be real numbers")
    try:
        margins = np.asarray(margins, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sensing margins must be a finite numeric vector") from exc
    if margins.ndim != 1 or margins.size == 0:
        raise ValueError("sensing margins must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(margins)) or np.any(margins < 0.0):
        raise ValueError("sensing margins must be finite and nonnegative")
    if expected_uavs is not None:
        if (isinstance(expected_uavs, (bool, np.bool_))
                or not isinstance(expected_uavs, (int, np.integer))
                or expected_uavs <= 0):
            raise ValueError("expected_uavs must be a positive integer")
        if margins.size != expected_uavs:
            raise ValueError("sensing margin count does not match the UAV count")

    scores = np.zeros_like(margins)
    scores[margins == 0.0] = 1.0
    shortfall = (margins > 0.0) & (margins < 1.0)
    deficits = -np.log(margins[shortfall])
    log_reference = np.log1p(reference_factor - 1.0)
    scores[shortfall] = deficits / (deficits + log_reference)
    return float(np.mean(scores)), scores
