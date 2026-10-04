"""Structured multiblock mesh.

* Each `Block` is a (possibly curvilinear) hexahedral structured block with `ng`
  ghost layers on every side.  Cell arrays have shape (..., Ni, Nj, Nk) with
  N = n + 2*ng, node arrays (3, N+1, ...).
* Blocks are joined with `Mesh.connect`.  The relative index orientation
  (axis permutation + flips) is detected automatically by matching the
  interface node planes, so blocks may be arbitrarily rotated w.r.t. each other.
  Periodicity is just a connection with a translation `shift` (a block may be
  connected to itself).
* Because the conserved variables use *physical* Cartesian momentum components,
  a halo exchange is a pure index remap - no vector rotation is required.
* All metric terms are computed geometrically (face area vectors + divergence
  theorem volumes), so the finite-volume scheme preserves a uniform free stream
  on curvilinear grids by construction.
"""
from itertools import product
from typing import NamedTuple

import numpy as np

from .backend import xp, asarray, set_at

FACE_NAMES = {"imin": (0, 0), "imax": (0, 1),
              "jmin": (1, 0), "jmax": (1, 1),
              "kmin": (2, 0), "kmax": (2, 1)}


def sl(nd, d, s, other=slice(None)):
    """Index tuple for an nd-array whose last 3 axes are spatial:
    `s` on spatial axis d, `other` on the remaining spatial axes."""
    lead = nd - 3
    t = [slice(None)] * lead + [other] * 3
    t[lead + d] = s
    return tuple(t)


class Geom(NamedTuple):
    """Static geometry of one block (a JAX pytree, so it can be a jit argument)."""
    vol: object      # (Ni,Nj,Nk)
    xc: object       # (3,Ni,Nj,Nk) cell centres
    S: tuple         # 3 arrays; S[d] has shape (3, ..N_d+1..) face area vectors (+d dir)
    xf: tuple        # 3 arrays, face centres, same shapes as S


# --------------------------------------------------------------------------- geometry
def _cross(a, b):
    return np.stack([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def compute_geometry(X):
    """Metrics from an (extended) node array X of shape (3, Ni+1, Nj+1, Nk+1)."""
    _, n1, n2, n3 = X.shape
    Ni, Nj, Nk = n1 - 1, n2 - 1, n3 - 1

    def face(get):
        c00, c10, c01, c11 = get(0, 0), get(1, 0), get(0, 1), get(1, 1)
        return 0.5 * _cross(c11 - c00, c01 - c10), 0.25 * (c00 + c10 + c01 + c11)

    # (first, second) tangent order is cyclic so that S points along +d
    Si, xi = face(lambda a, b: X[:, :, a:Nj + a, b:Nk + b])        # (j,k)
    Sj, xj = face(lambda a, b: X[:, b:Ni + b, :, a:Nk + a])        # (k,i)
    Sk, xk = face(lambda a, b: X[:, a:Ni + a, b:Nj + b, :])        # (i,j)

    dot = lambda a, b: np.sum(a * b, axis=0)
    vol = (dot(xi[:, 1:], Si[:, 1:]) - dot(xi[:, :-1], Si[:, :-1])
           + dot(xj[:, :, 1:], Sj[:, :, 1:]) - dot(xj[:, :, :-1], Sj[:, :, :-1])
           + dot(xk[..., 1:], Sk[..., 1:]) - dot(xk[..., :-1], Sk[..., :-1])) / 3.0
    xc = sum(X[:, a:Ni + a, b:Nj + b, c:Nk + c]
             for a, b, c in product((0, 1), repeat=3)) / 8.0
    return vol, xc, (Si, Sj, Sk), (xi, xj, xk)


# --------------------------------------------------------------------------- blocks
class Interface:
    """Block-to-block connection seen from the *destination* block."""

    def __init__(self, nb, src_face, mapping, flip, shift):
        self.nb = nb                  # index of neighbour (source) block
        self.src_face = src_face      # (axis, side) on the neighbour
        self.mapping = mapping        # mapping[A axis] -> B axis
        self.flip = flip              # flip[A axis] (tangential only)
        self.shift = np.asarray(shift, float)   # x_A(ghost) = x_B + shift


class Block:
    def __init__(self, name, nodes, ng=3):
        nodes = np.asarray(nodes, float)
        assert nodes.ndim == 4 and nodes.shape[0] == 3, "nodes must be (3,ni+1,nj+1,nk+1)"
        self.name, self.ng, self.x_int = name, ng, nodes
        self.n = tuple(s - 1 for s in nodes.shape[1:])
        assert min(self.n) >= ng, "each block needs at least ng cells per direction"
        self.N = tuple(n + 2 * ng for n in self.n)
        self.faces = {}
        self.X = None
        self.geom = None

    def set_bc(self, face, bc):
        self.faces[FACE_NAMES[face]] = bc

    # ---- slicing helpers (work for any array with 3 trailing spatial axes)
    def interior(self, a):
        lead, ng = a.ndim - 3, self.ng
        return a[(slice(None),) * lead + tuple(slice(ng, ng + n) for n in self.n)]

    def take(self, a, axis, idx):
        """Layer `idx` (int, full-array index) along `axis`; interior tangential range."""
        lead, ng = a.ndim - 3, self.ng
        t = [slice(None)] * lead + [slice(ng, ng + n) for n in self.n]
        t[lead + axis] = idx
        return a[tuple(t)]

    def slab(self, a, d, lo, hi):
        """Cells ng+lo .. ng+n_d+hi along d (interior tangential).  slab(-1,0) are the
        'left' cells of the n_d+1 interior faces of direction d, slab(0,1) the 'right' ones."""
        lead, ng = a.ndim - 3, self.ng
        t = [slice(None)] * lead + [slice(ng, ng + n) for n in self.n]
        t[lead + d] = slice(ng + lo, ng + self.n[d] + hi)
        return a[tuple(t)]


# --------------------------------------------------------------------------- exchange
def exchange(dst, src, blkA, faceA, blkB, itf, kind, inplace=False):
    """Fill the ghost layers of `dst` (block A, face faceA) from `src` (block B).
    kind='cell' : cell-centred data, ghost layer m <- B's m-th cell off the interface
    kind='node' : node data (adds itf.shift); the interface node plane is shared."""
    aA, sA = faceA
    aB, sB = itf.src_face
    ng, nA, nB = blkA.ng, blkA.n[aA], blkB.n[aB]
    lead = dst.ndim - 3
    lib = np if isinstance(dst, np.ndarray) and inplace else xp
    if kind == "cell":
        lo_src = ng if sB == 0 else nB
        lo_dst = ng + nA if sA == 1 else 0
    else:
        lo_src = ng + 1 if sB == 0 else nB
        lo_dst = ng + nA + 1 if sA == 1 else 0
    S = src[sl(src.ndim, aB, slice(lo_src, lo_src + ng))]
    if sB == 1:
        S = lib.flip(S, axis=lead + aB)                        # -> m-order
    S = lib.transpose(S, list(range(lead)) + [lead + itf.mapping[m] for m in range(3)])
    for t in range(3):
        if itf.flip[t]:
            S = lib.flip(S, axis=lead + t)
    if kind == "node":
        S = S + itf.shift.reshape(3, 1, 1, 1)
    if sA == 0:
        S = lib.flip(S, axis=lead + aA)                        # m-order -> position
    idx = sl(dst.ndim, aA, slice(lo_dst, lo_dst + ng))
    if inplace:
        dst[idx] = S
        return dst
    return set_at(dst, idx, S)


def _reflect_nodes(X, blk, axis, side):
    ng, n = blk.ng, blk.n[axis]
    if side == 0:
        g = 2 * X[sl(4, axis, slice(ng, ng + 1))] - np.flip(X[sl(4, axis, slice(ng + 1, ng + 1 + ng))], axis=1 + axis)
        X[sl(4, axis, slice(0, ng))] = g
    else:
        b = ng + n
        g = 2 * X[sl(4, axis, slice(b, b + 1))] - np.flip(X[sl(4, axis, slice(n, n + ng))], axis=1 + axis)
        X[sl(4, axis, slice(b + 1, b + 1 + ng))] = g


def _match_orientation(A, fa, B, fb, shift):
    """Find axis mapping/flips such that B's interface node plane (+shift) equals A's."""
    aA, sA = fa
    aB, sB = fb
    plane = lambda blk, a, s: np.take(blk.x_int, 0 if s == 0 else blk.n[a], axis=1 + a)
    pa = plane(A, aA, sA)
    pb = plane(B, aB, sB) + np.asarray(shift, float).reshape(3, 1, 1)
    tA = [m for m in range(3) if m != aA]
    tB = [m for m in range(3) if m != aB]
    scale = max(np.abs(pa).max(), 1e-30)
    for perm in ([tB[0], tB[1]], [tB[1], tB[0]]):
        pbt = np.transpose(pb, [0] + [1 + tB.index(p) for p in perm])
        for f1, f2 in product((0, 1), repeat=2):
            c = pbt
            if f1:
                c = np.flip(c, 1)
            if f2:
                c = np.flip(c, 2)
            if c.shape == pa.shape and np.allclose(c, pa, atol=1e-9 * scale, rtol=0):
                mapping, flip = [0] * 3, [False] * 3
                mapping[aA], mapping[tA[0]], mapping[tA[1]] = aB, perm[0], perm[1]
                flip[tA[0]], flip[tA[1]] = bool(f1), bool(f2)
                return tuple(mapping), tuple(flip)
    raise ValueError(f"interface {A.name}{fa} <-> {B.name}{fb}: node planes do not match "
                     f"(check the shift vector and the block node coordinates)")


class Mesh:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        assert len({b.ng for b in self.blocks}) == 1, "all blocks must share ng"
        self.ng = self.blocks[0].ng

    def index(self, blk):
        return next(i for i, b in enumerate(self.blocks) if b is blk)

    def connect(self, A, faceA, B, faceB, shift=(0.0, 0.0, 0.0)):
        """Join A.faceA to B.faceB.  `shift` = x_A(plane) - x_B(plane) (periodic offset)."""
        fa, fb = FACE_NAMES[faceA], FACE_NAMES[faceB]
        shift = np.asarray(shift, float)
        for X, fx, Y, fy, sh in ((A, fa, B, fb, shift), (B, fb, A, fa, -shift)):
            mapping, flip = _match_orientation(X, fx, Y, fy, sh)
            X.faces[fx] = Interface(self.index(Y), fy, mapping, flip, sh)

    def finalize(self, node_passes=3):
        ng = self.ng
        for b in self.blocks:
            for f in FACE_NAMES.values():
                if f not in b.faces:
                    raise KeyError(f"block {b.name}: face {f} has no BC / connection")
            X = np.zeros((3,) + tuple(n + 1 + 2 * ng for n in b.n))
            X[(slice(None),) + tuple(slice(ng, ng + n + 1) for n in b.n)] = b.x_int
            b.X = X
        # ghost nodes: several sweeps so edge/corner ghosts across blocks converge
        for _ in range(node_passes):
            for a in range(3):
                for b in self.blocks:
                    for s in (0, 1):
                        f = b.faces[(a, s)]
                        if isinstance(f, Interface):
                            exchange(b.X, self.blocks[f.nb].X, b, (a, s), self.blocks[f.nb], f,
                                     "node", inplace=True)
                        else:
                            _reflect_nodes(b.X, b, a, s)
        geoms = []
        for b in self.blocks:
            vol, xc, S, xf = compute_geometry(b.X)
            interior = vol[ng:ng + b.n[0], ng:ng + b.n[1], ng:ng + b.n[2]]
            if not np.isfinite(vol).all() or np.any(vol == 0) or np.any(interior <= 0):
                raise ValueError(f"block {b.name}: non-positive physical or degenerate ghost-cell volume")
            b.geom = Geom(asarray(vol), asarray(xc),
                          tuple(asarray(s) for s in S), tuple(asarray(x) for x in xf))
            geoms.append(b.geom)
        self.geoms = tuple(geoms)
        return self
