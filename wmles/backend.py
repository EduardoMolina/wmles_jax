"""Array backend switch.

JAX (float64) is used when available. Set WMLES_BACKEND=numpy to force NumPy,
which is handy for debugging with a plain debugger / no accelerator.
Every module writes purely functional array code (no in-place mutation), so the
same source runs under both backends; `set_at` hides the only difference.
"""
import os

_FORCE_NUMPY = os.environ.get("WMLES_BACKEND", "").lower() == "numpy"
_DEVICE_REQUEST = os.environ.get("WMLES_DEVICE", "auto").strip().lower()
if _DEVICE_REQUEST not in {"auto", "cpu", "gpu", "cuda"}:
    raise ValueError("WMLES_DEVICE must be one of: auto, cpu, gpu, cuda")

if _FORCE_NUMPY and _DEVICE_REQUEST in {"gpu", "cuda"}:
    raise RuntimeError("WMLES_DEVICE=gpu requires the JAX backend; unset WMLES_BACKEND=numpy")

try:
    if _FORCE_NUMPY:
        raise ImportError
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as xp
    _PLATFORM = "cpu" if _DEVICE_REQUEST == "cpu" else "gpu" if _DEVICE_REQUEST in {"gpu", "cuda"} else None
    try:
        _DEVICES = jax.devices(_PLATFORM) if _PLATFORM else jax.devices()
    except RuntimeError as exc:
        if _PLATFORM == "gpu":
            raise RuntimeError("WMLES_DEVICE=gpu requested, but JAX could not initialize a GPU device") from exc
        raise
    if not _DEVICES:
        requested = _PLATFORM or "default"
        raise RuntimeError(f"JAX reported no {requested} devices")
    DEVICE = _DEVICES[0]
    HAS_JAX = True
except ImportError as exc:  # pragma: no cover
    if _DEVICE_REQUEST in {"gpu", "cuda"}:
        raise RuntimeError("WMLES_DEVICE=gpu requested, but JAX is not installed") from exc
    import numpy as xp
    DEVICE = None
    HAS_JAX = False


def device_info():
    """Describe the selected array backend and device."""
    if HAS_JAX:
        return f"jax:{DEVICE.platform}:{DEVICE}"
    return "numpy:cpu"


def asarray(value, dtype=None):
    """Convert an array and place it on the selected JAX device when enabled."""
    array = xp.asarray(value, dtype=dtype)
    return jax.device_put(array, DEVICE) if HAS_JAX else array


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
