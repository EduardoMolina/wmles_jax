import os

# Virtual CPU devices let the distributed-solver tests run without GPUs. This must be
# set before JAX initialises, which importing any module of this package guarantees.
_FLAG = "--xla_force_host_platform_device_count"
if _FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + f" {_FLAG}=8").strip()
