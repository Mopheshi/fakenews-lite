"""Deterministic seeding across Python, NumPy and TensorFlow.

Call :func:`set_global_seed` before building features or models. Bitwise
reproducibility on CPU additionally requires ``PYTHONHASHSEED`` to be set in the
environment *before* the interpreter starts, which :func:`hashseed_is_fixed`
lets callers verify.
"""

from __future__ import annotations

import os
import random

import numpy as np


def hashseed_is_fixed() -> bool:
    """Return True when PYTHONHASHSEED was pinned before interpreter start."""
    return os.environ.get("PYTHONHASHSEED") not in (None, "random")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow (including Keras initialisers)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:  # TensorFlow is optional for TF-IDF-only workflows.
        return

    tf.keras.utils.set_random_seed(seed)


def enable_deterministic_ops() -> None:
    """Force deterministic TensorFlow kernels.

    Measurably slower, so it is opt-in rather than part of ``set_global_seed``.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    try:
        import tensorflow as tf
    except ImportError:
        return

    tf.config.experimental.enable_op_determinism()
