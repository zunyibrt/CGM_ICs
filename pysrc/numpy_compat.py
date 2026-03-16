"""Compatibility helpers for third-party libraries pinned below NumPy 2."""

import numpy as np
from numpy.lib import _function_base_impl


if not hasattr(np, "in1d"):
    np.in1d = np.isin


if not hasattr(_function_base_impl, "_check_interpolation_as_method"):
    def _check_interpolation_as_method(method, interpolation, fname):
        if interpolation is None:
            return method
        if method not in (None, "linear", interpolation):
            return method
        return interpolation

    _function_base_impl._check_interpolation_as_method = _check_interpolation_as_method
