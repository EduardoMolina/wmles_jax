"""Cell-centred Green-Gauss gradients and viscous face fluxes."""
from .backend import xp
from .mesh import sl


def green_gauss(phi, geom):
    """phi: (nv, Ni, Nj, Nk) -> gradient (nv, 3, Ni, Nj, Nk); outermost cell layer = 0."""
    nd = phi.ndim
    acc = 0.0
    for d in range(3):
        Nd = phi.shape[1 + d]
        pf = 0.5 * (phi[sl(nd, d, slice(None, -1))] + phi[sl(nd, d, slice(1, None))])
        Sf = geom.S[d][sl(4, d, slice(1, Nd))]                      # faces 1..N_d-1
        flux = pf[:, None] * Sf[None]                               # (nv,3,...)
        inner = slice(1, -1)
        acc = acc + (flux[sl(nd + 1, d, slice(1, None), inner)]
                     - flux[sl(nd + 1, d, slice(None, -1), inner)])
    g = acc / geom.vol[1:-1, 1:-1, 1:-1]
    return xp.pad(g, [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)])


def ducros(gu):
    div = gu[0, 0] + gu[1, 1] + gu[2, 2]
    w2 = ((gu[2, 1] - gu[1, 2]) ** 2 + (gu[0, 2] - gu[2, 0]) ** 2 + (gu[1, 0] - gu[0, 1]) ** 2)
    return div ** 2 / (div ** 2 + w2 + 1e-30)


def viscous_flux(blk, geom, d, phi, g, mu_eff, k_eff):
    """Viscous flux through the n_d+1 interior faces of direction d (times face area).
    phi = [u,v,w,T] (4,..), g = grad(phi) (4,3,..).  Face gradient = arithmetic mean of the
    two cell gradients + compact (two-cell) correction along the cell-to-cell vector."""
    Lf = lambda a: blk.slab(a, d, -1, 0)
    Rf = lambda a: blk.slab(a, d, 0, 1)
    gbar = 0.5 * (Lf(g) + Rf(g))
    dx = Rf(geom.xc) - Lf(geom.xc)
    dist = xp.sqrt(xp.sum(dx * dx, axis=0))
    e = dx / dist
    corr = (Rf(phi) - Lf(phi)) / dist - xp.sum(gbar * e[None], axis=1)
    gf = gbar + corr[:, None] * e[None]
    uf = 0.5 * (Lf(phi[:3]) + Rf(phi[:3]))
    muf = 0.5 * (Lf(mu_eff) + Rf(mu_eff))
    kf = 0.5 * (Lf(k_eff) + Rf(k_eff))
    S = blk.slab(geom.S[d], d, 0, 1)
    gu = gf[:3]
    div = gu[0, 0] + gu[1, 1] + gu[2, 2]
    eye = xp.eye(3).reshape(3, 3, 1, 1, 1)
    tau = muf * (gu + xp.swapaxes(gu, 0, 1) - (2.0 / 3.0) * eye * div)
    tauS = xp.einsum('ij...,j...->i...', tau, S)
    dTdn = xp.einsum('j...,j...->...', gf[3], S)
    return xp.concatenate([xp.zeros_like(div)[None], -tauS,
                           (-xp.sum(uf * tauS, axis=0) - kf * dTdn)[None]], axis=0)
