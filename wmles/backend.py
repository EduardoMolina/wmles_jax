"""Array backend switch.

JAX (float64) is used when available. Set WMLES_BACKEND=numpy to force NumPy,
which is handy for debugging with a plain debugger / no accelerator.
Every module writes purely functional array code (no in-place mutation), so the
same source runs under both backends; `set_at` hides the only difference.

Multi-device runs: WMLES_NUM_DEVICES=N asks for N devices of the selected platform
(default 1).  `select_devices` never falls back silently: asking for more devices
than JAX exposes raises.  `python -m wmles.backend --require 8` is the AWS gate.
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
    _DEVICES = []
    DEVICE = None
    HAS_JAX = False


def requested_device_count():
    """Device count from WMLES_NUM_DEVICES (default 1)."""
    text = os.environ.get("WMLES_NUM_DEVICES", "1").strip()
    try:
        count = int(text)
    except ValueError:
        raise ValueError(f"WMLES_NUM_DEVICES must be a positive integer, got {text!r}") from None
    if count < 1:
        raise ValueError(f"WMLES_NUM_DEVICES must be a positive integer, got {text!r}")
    return count


def select_devices(count=None):
    """The first `count` devices of the selected platform (default WMLES_NUM_DEVICES).

    Raises instead of returning fewer devices, so a multi-GPU run cannot quietly
    degrade to one GPU or to the CPU."""
    if not HAS_JAX:
        raise RuntimeError("multi-device execution requires the JAX backend")
    count = requested_device_count() if count is None else int(count)
    if count < 1:
        raise ValueError("device count must be positive")
    if count > len(_DEVICES):
        raise RuntimeError(
            f"{count} devices requested but only {len(_DEVICES)} "
            f"{_DEVICES[0].platform} device(s) are visible: {_DEVICES}"
        )
    return tuple(_DEVICES[:count])


def device_info():
    """Describe the selected array backend and device."""
    if HAS_JAX:
        return f"jax:{DEVICE.platform}:{DEVICE}"
    return "numpy:cpu"


def asarray(value, dtype=None, device=None):
    """Convert an array and place it on `device` (default: the selected device) when JAX is enabled."""
    array = xp.asarray(value, dtype=dtype)
    return jax.device_put(array, device or DEVICE) if HAS_JAX else array


def device_put(value, device):
    """Commit `value` (array or pytree) to `device`; a no-op without JAX."""
    return jax.device_put(value, device) if HAS_JAX else value


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


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Report the JAX devices visible to wmles.")
    parser.add_argument("--require", type=int, default=0, metavar="N",
                        help="exit non-zero unless at least N devices of the selected platform exist")
    args = parser.parse_args(argv)
    print(f"backend: {device_info()}")
    for device in _DEVICES:
        print(f"  {device.platform}:{device.id} {device}")
    if args.require:
        try:
            select_devices(args.require)
        except RuntimeError as exc:
            print(f"FAIL: {exc}")
            return 1
        print(f"OK: {args.require} {_DEVICES[0].platform} devices available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
