"""Sod shock tube through the MUSCL-HLLC / Hybrid paths + conservation check.
   python -m tests.test_sod"""
import numpy as np
from wmles import *
from wmles.backend import xp, HAS_JAX

def run(scheme, nx=100, tend=0.2):
    gas = Gas(R=1.0, mu_ref=1e-9)
    x = np.linspace(0, 1, nx + 1); y = np.linspace(0, 0.1, 4); z = np.linspace(0, 0.1, 4)
    b = Block("tube", np.stack(np.meshgrid(x, y, z, indexing="ij")))
    m = Mesh([b])
    b.set_bc("imin", Extrapolate()); b.set_bc("imax", Extrapolate())
    m.connect(b, "jmin", b, "jmax", shift=(0, -0.1, 0))    # periodic: plane spacing = 0.1
    m.connect(b, "kmin", b, "kmax", shift=(0, 0, -0.1))
    m.finalize()
    s = Solver(m, gas, scheme, sgs=None, cfl=0.5)
    prim = lambda xc: xp.stack([xp.where(xc[0] < 0.5, 1.0, 0.125), 0 * xc[0], 0 * xc[0], 0 * xc[0],
                                xp.where(xc[0] < 0.5, 1.0, 0.1)])
    Q = s.init_state(prim)
    mass0 = float(xp.sum(b.interior(Q[0]) * b.interior(m.geoms[0].vol)))
    t = 0.0
    while t < tend - 1e-12:
        dt = min(float(s.max_dt(Q)), tend - t)
        Q = s.advance(Q, dt); t += dt
    w, _ = gas.cons_to_prim(Q[0])
    w = np.asarray(b.interior(w))[:, :, 1, 1]
    xc = np.asarray(b.interior(m.geoms[0].xc))[0, :, 1, 1]
    return xc, w

if __name__ == "__main__":
    for sc in (MUSCLHLLC(), Hybrid(th0=0.3, th1=0.6)):
        xc, w = run(sc)
        i = np.argmin(abs(xc - 0.75))       # between contact (x~0.685) and shock (x~0.850)
        j = np.argmin(abs(xc - 0.60))       # between rarefaction tail and contact
        print(f"{type(sc).__name__:10s} p(0.75)={w[4, i]:.4f} (exact 0.3031)  u(0.75)={w[1, i]:.4f} (exact 0.9275)  "
              f"rho(0.75)={w[0, i]:.4f} (exact 0.2656)  rho(0.60)={w[0, j]:.4f} (exact 0.4263)")
