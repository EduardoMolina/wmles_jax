"""Array backend switch.

JAX (float64) is used when available. Set WMLES_BACKEND=numpy to force NumPy,
which is handy for debugging with a plain debugger / no accelerator.
Every module writes purely functional array code (no in-place mutation), so the
same source runs under both backends; `set_at` hides the only difference.
"""
import os

_FORCE_NUMPY = os.environ.get("WMLES_BACKEND", "").lower() == "numpy"
try:
    if _FORCE_NUMPY:
        raise ImportError
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as xp
    HAS_JAX = True
except ImportError:  # pragma: no cover
    import numpy as xp
    HAS_JAX = False


def set_at(a, idx, v):
    """Functional `a[idx] = v` (returns a new array)."""
    if HAS_JAX:
        return a.at[idx].set(v)
    a = a.copy()
    a[idx] = v
    return a


def jit(fn, **kw):
    return jax.jit(fn, **kw) if HAS_JAX else fn


def fori_loop(lo, hi, body, init):
    if HAS_JAX:
        return jax.lax.fori_loop(lo, hi, body, init)
    x = init
    for i in range(lo, hi):
        x = body(i, x)
    return x
