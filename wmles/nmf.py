"""Neutral Map File (NMF) support and multiblock mesh construction from Plot3D grids.

    from wmles.plot3d import read_plot3d
    from wmles.nmf import read_nmf, respan, build_mesh, describe

    grid = read_plot3d("nak_a_ff_500c_2zn_L5.p3dfmt")
    nmf  = read_nmf("nak_a_ff_500c_2zn_L5.nmf")
    nodes, nmf = respan(grid.nodes, nmf, ncells=8, length=0.1)      # 1 spanwise cell -> 8
    mesh = build_mesh(nodes, nmf,
                      bc_map={"viscous_solid": lambda: WallModel(...),
                              "farfield_riem": Dirichlet(prim)},
                      periodic=(("symmetry_y", 0),))                # spanwise periodic
    print(describe(mesh))

NMF face convention (NASA GEOLAB; https://tmbwg.github.io/turbmodels/nmf_documentation.html):
    face  constant index   primary, secondary axis
     1    kmin             i, j
     2    kmax             i, j
     3    imin             j, k
     4    imax             j, k
     5    jmin             k, i
     6    jmax             k, i
S1..E1 / S2..E2 are 1-based *node* ranges along the primary/secondary axis.  For a
ONE_TO_ONE entry the second block's ranges may run backwards (that encodes orientation);
Swap=TRUE means the primary axis of block 1 maps to the secondary axis of block 2.

Sub-face patches
----------------
`Mesh.connect` joins a *whole* block face to a *whole* block face. NMF files, however, may
connect one face to several partners (the TMBWG airfoil wake grids do: the wake block's
face is joined half to the airfoil block's jmax face and half to its jmin face).
`build_mesh` therefore splits blocks along every patch boundary (propagating the cuts
through one-to-one interfaces until nothing changes) so that each sub-block face is exactly
one patch; the new inner planes become ordinary interfaces. Orientation of every interface
is then re-derived - and thereby verified - from the node coordinates by `Mesh.connect`.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

from .mesh import Block, Mesh, Interface
from .bc import BC, SlipWall, NoSlipWall

# NMF face number -> (normal axis, side, (primary axis, secondary axis))
FACES = {1: (2, 0, (0, 1)), 2: (2, 1, (0, 1)),
         3: (0, 0, (1, 2)), 4: (0, 1, (1, 2)),
         5: (1, 0, (2, 0)), 6: (1, 1, (2, 0))}
FACE_NAME = {(0, 0): "imin", (0, 1): "imax", (1, 0): "jmin", (1, 1): "jmax",
             (2, 0): "kmin", (2, 1): "kmax"}

# NMF boundary-type name (lower case, '-' -> '_') -> BC. Far-field / inflow / outflow types are
# deliberately absent: supply them through `bc_map`.
DEFAULT_BC_MAP = {
    "symmetry_x": SlipWall, "symmetry_y": SlipWall, "symmetry_z": SlipWall,
    "wall": NoSlipWall, "viscous_solid": NoSlipWall,
    "inviscid_solid": SlipWall,
}

_UNSUPPORTED = ("collapsed", "pole_dir1", "pole_dir2", "patched", "unprocessed")


@dataclass
class Patch:
    type: str
    block: int               # 0-based
    face: int                # 1..6
    ranges: tuple            # ((S1,E1),(S2,E2)), 1-based inclusive, as in the file
    partner: object = None   # None, or dict(block, face, ranges, swap) for one_to_one


@dataclass
class NMF:
    dims: list               # [(ni,nj,nk)] node counts
    patches: list


def read_nmf(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].replace("\\", " ").strip()
            if line:
                rows.append(line.split())
    nb = int(rows[0][0])
    dims = [None] * nb
    for r in rows[1: 1 + nb]:
        b, i, j, k = (int(t) for t in r[:4])
        dims[b - 1] = (i, j, k)
    patches = []
    for r in rows[1 + nb:]:
        typ = r[0].lower().replace("-", "_")
        if typ in _UNSUPPORTED:
            raise NotImplementedError(f"NMF entry type '{r[0]}' is not supported "
                                      f"(only one-to-one interfaces and boundary conditions)")
        args = r[1:]
        if typ == "one_to_one":
            v = [int(t) for t in args[:12]]
            swap = len(args) > 12 and args[12].lower() in ("true", ".true.", "t", "1")
            patches.append(Patch(typ, v[0] - 1, v[1], ((v[2], v[3]), (v[4], v[5])),
                                 dict(block=v[6] - 1, face=v[7], ranges=((v[8], v[9]), (v[10], v[11])),
                                      swap=swap)))
        else:
            v = [int(t) for t in args[:6]]
            patches.append(Patch(typ, v[0] - 1, v[1], ((v[2], v[3]), (v[4], v[5]))))
    return NMF(dims, patches)


# ---------------------------------------------------------------------------- span handling
def respan(nodes, nmf, ncells, axis=0, length=None):
    """Re-discretize a block axis that has exactly two node planes (a 'one spanwise cell' grid such as
    the TMBWG airfoil grids, span = axis 0) into `ncells` uniform cells, optionally rescaling the total
    span to `length`. All blocks must use `axis` as the span and share the same span vector.
    Returns (new_nodes, new_nmf); inputs are not modified."""
    D = None
    out = []
    for b, X in enumerate(nodes):
        if X.shape[1 + axis] != 2:
            raise ValueError(f"block {b + 1}: axis {axis} has {X.shape[1 + axis]} nodes, expected 2")
        X0, X1 = np.take(X, 0, axis=1 + axis), np.take(X, 1, axis=1 + axis)
        d = X1 - X0
        dv = d.reshape(3, -1)[:, 0]
        if not np.allclose(d, dv.reshape((3,) + (1,) * (d.ndim - 1)), atol=1e-9 * max(np.abs(X).max(), 1.0)):
            raise ValueError(f"block {b + 1}: the two planes along axis {axis} are not a pure translation")
        if D is None:
            D = dv
        elif not np.allclose(dv, D, atol=1e-9 * max(np.linalg.norm(D), 1.0)):
            raise ValueError("blocks disagree on the span vector")
    Dn = D * (length / np.linalg.norm(D)) if length else D
    for X in nodes:
        X0 = np.take(X, 0, axis=1 + axis)
        out.append(np.stack([X0 + (m / ncells) * Dn.reshape(3, 1, 1) for m in range(ncells + 1)], axis=1 + axis))
    return out, _renumber_axis(nmf, axis, 2, ncells + 1)


def _renumber_axis(nmf, axis, old, new):
    dims = [tuple(new if a == axis else n for a, n in enumerate(d)) for d in nmf.dims]

    def fix(face, ranges):
        _, _, axes = FACES[face]
        res = []
        for ax, (s, e) in zip(axes, ranges):
            if ax == axis:
                if {s, e} != {1, old}:
                    raise ValueError(f"cannot respan: a patch covers only part of axis {axis}")
                s, e = (1, new) if s < e else (new, 1)
            res.append((s, e))
        return tuple(res)

    patches = []
    for p in nmf.patches:
        q = Patch(p.type, p.block, p.face, fix(p.face, p.ranges), None)
        if p.partner:
            q.partner = dict(p.partner, ranges=fix(p.partner["face"], p.partner["ranges"]))
        patches.append(q)
    return NMF(dims, patches)


def extrude_2d(nodes, ncells, length, direction=(0.0, 0.0, 1.0)):
    """(3,ni,nj,1) blocks from a 2-D Plot3D file -> (3,ni,nj,ncells+1) blocks extruded along `direction`
    (new k axis). Note: NMF files written for 3-D grids do not apply to the result."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d) * length
    return [np.stack([X[..., 0] + (m / ncells) * d.reshape(3, 1, 1) for m in range(ncells + 1)], axis=3)
            for X in nodes]


# ---------------------------------------------------------------------------- topology
def _tangential(face, ranges):
    """[(axis, lo, hi)] 0-based, lo<=hi for the two tangential axes of an NMF face patch."""
    _, _, axes = FACES[face]
    return [(ax, min(s, e) - 1, max(s, e) - 1) for ax, (s, e) in zip(axes, ranges)]


def _axis_pairs(p):
    """One-to-one correspondence per tangential axis: [(axisA,(a_s,a_e), axisB,(b_s,b_e))], 0-based,
    directions as written in the file."""
    B = p.partner
    _, _, (pa, qa) = FACES[p.face]
    _, _, (pb, qb) = FACES[B["face"]]
    rA = [(s - 1, e - 1) for s, e in p.ranges]
    rB = [(s - 1, e - 1) for s, e in B["ranges"]]
    pairs = ([(pa, rA[0], qb, rB[1]), (qa, rA[1], pb, rB[0])] if B["swap"]
             else [(pa, rA[0], pb, rB[0]), (qa, rA[1], qb, rB[1])])
    for ax_a, (a0, a1), ax_b, (b0, b1) in pairs:
        if abs(a1 - a0) != abs(b1 - b0):
            raise ValueError(f"inconsistent one_to_one ranges: block {p.block + 1} face {p.face} vs "
                             f"block {B['block'] + 1} face {B['face']}")
    return pairs


def _map(m, a_rng, b_rng):
    (a0, a1), (b0, b1) = a_rng, b_rng
    if a1 == a0:
        return b0
    return b0 + (m - a0) * (b1 - b0) // (a1 - a0)


def _resolve_bc(bc_map, typ):
    v = bc_map.get(typ, DEFAULT_BC_MAP.get(typ))
    if v is None:
        raise KeyError(f"no BC supplied for NMF type '{typ}' - pass bc_map={{'{typ}': <BC or factory>}}")
    return v if isinstance(v, BC) else v()


def build_mesh(nodes, nmf, bc_map=None, periodic=(), ng=3, finalize=True):
    """Assemble a `Mesh` (with automatic sub-block splitting, see module docstring).

    bc_map    {nmf_type: BC instance or zero-argument factory}; extends/overrides DEFAULT_BC_MAP
    periodic  iterable of (nmf_type, axis): patches of that type on faces normal to `axis` are turned into a
              periodic connection between each block's min/max faces (translation taken from the nodes)
    Each returned block has `.origin = dict(block=<0-based NMF block>, ranges=((i0,i1),(j0,j1),(k0,k1)))`,
    the node range it occupies in the original block."""
    bc_map = dict(bc_map or {})
    periodic = {(t.lower().replace("-", "_"), ax) for t, ax in periodic}
    nb = len(nodes)
    for b in range(nb):
        if tuple(nodes[b].shape[1:]) != tuple(nmf.dims[b]):
            raise ValueError(f"block {b + 1}: grid has {nodes[b].shape[1:]} nodes, NMF says {nmf.dims[b]}")

    # 1. cut planes from every patch boundary, propagated through one-to-one interfaces
    cuts = [[set(), set(), set()] for _ in range(nb)]
    for p in nmf.patches:
        sides = [(p.block, p.face, p.ranges)]
        if p.partner:
            sides.append((p.partner["block"], p.partner["face"], p.partner["ranges"]))
        for b, face, ranges in sides:
            for ax, lo, hi in _tangential(face, ranges):
                n = nmf.dims[b][ax]
                if lo > 0:
                    cuts[b][ax].add(lo)
                if hi < n - 1:
                    cuts[b][ax].add(hi)
    o2o = [p for p in nmf.patches if p.partner]
    changed = True
    while changed:
        changed = False
        for p in o2o:
            for ax_a, ra, ax_b, rb in _axis_pairs(p):
                A, B = p.block, p.partner["block"]
                for src_b, src_ax, src_r, dst_b, dst_ax, dst_r in ((A, ax_a, ra, B, ax_b, rb),
                                                                    (B, ax_b, rb, A, ax_a, ra)):
                    lo, hi = sorted(src_r)
                    for m in [c for c in cuts[src_b][src_ax] if lo < c < hi]:
                        m2 = _map(m, src_r, dst_r)
                        if m2 not in cuts[dst_b][dst_ax]:
                            cuts[dst_b][dst_ax].add(m2)
                            changed = True

    # 2. sub-blocks
    bounds = [[[0] + sorted(cuts[b][ax]) + [nmf.dims[b][ax] - 1] for ax in range(3)] for b in range(nb)]
    subs = {}                                    # (b, (ia,ib,ic)) -> Block
    order = []
    for b in range(nb):
        nseg = [len(bounds[b][ax]) - 1 for ax in range(3)]
        for idx in product(*(range(n) for n in nseg)):
            rng = tuple((bounds[b][ax][idx[ax]], bounds[b][ax][idx[ax] + 1]) for ax in range(3))
            X = nodes[b][:, rng[0][0]:rng[0][1] + 1, rng[1][0]:rng[1][1] + 1, rng[2][0]:rng[2][1] + 1]
            n_cells = tuple(r[1] - r[0] for r in rng)
            if min(n_cells) < ng:
                raise ValueError(f"block {b + 1}{list(idx)} has only {n_cells} cells; every direction needs >= ng={ng}"
                                 f" (a 1-cell-thick span must be widened first: see respan())")
            name = f"b{b + 1}" if max(nseg) == 1 else f"b{b + 1}_" + "".join(map(str, idx))
            blk = Block(name, X, ng=ng)
            blk.origin = dict(block=b, ranges=rng)
            subs[(b, idx)] = blk
            order.append((b, idx))
    mesh = Mesh([subs[k] for k in order])

    def face_on_boundary(b, idx, ax, side):
        rng = subs[(b, idx)].origin["ranges"][ax]
        return rng[side] == (0 if side == 0 else nmf.dims[b][ax] - 1)

    def within(sub_rng, tang):
        return all(lo <= sub_rng[ax][0] and sub_rng[ax][1] <= hi for ax, lo, hi in tang)

    # 3. inner planes created by the cuts -> ordinary interfaces
    for (b, idx), blk in subs.items():
        for ax in range(3):
            nxt = idx[:ax] + (idx[ax] + 1,) + idx[ax + 1:]
            if (b, nxt) in subs:
                mesh.connect(blk, FACE_NAME[(ax, 1)], subs[(b, nxt)], FACE_NAME[(ax, 0)])

    # 4. original faces: boundary conditions and one-to-one interfaces
    periodic_marks = set()
    for p in nmf.patches:
        ax, side, _ = FACES[p.face]
        tang = _tangential(p.face, p.ranges)
        for (b, idx), blk in subs.items():
            if b != p.block or not face_on_boundary(b, idx, ax, side) or not within(blk.origin["ranges"], tang):
                continue
            fkey = FACE_NAME[(ax, side)]
            if p.partner is None:
                if (p.type, ax) in periodic:
                    periodic_marks.add(((b, idx), ax, side))
                else:
                    blk.set_bc(fkey, _resolve_bc(bc_map, p.type))
                continue
            if (ax, side) in blk.faces:              # connection already made from the other side
                continue
            B = p.partner
            bax, bside, _ = FACES[B["face"]]
            want = {}
            for ax_a, ra, ax_b, rb in _axis_pairs(p):
                lo, hi = blk.origin["ranges"][ax_a]
                want[ax_b] = tuple(sorted((_map(lo, ra, rb), _map(hi, ra, rb))))
            match = [k for k, sb in subs.items()
                     if k[0] == B["block"] and face_on_boundary(k[0], k[1], bax, bside)
                     and all(sb.origin["ranges"][a] == r for a, r in want.items())]
            if len(match) != 1:
                raise ValueError(f"block {p.block + 1} face {p.face}: partner sub-face not found ({len(match)})")
            mesh.connect(blk, fkey, subs[match[0]], FACE_NAME[(bax, bside)])

    # 5. periodic pairs (both faces of the same sub-block)
    done = set()
    for (key, ax, side) in sorted(periodic_marks):
        if (key, ax) in done:
            continue
        if ((key, ax, 1 - side) not in periodic_marks):
            raise ValueError(f"{subs[key].name}: periodic axis {ax} needs the periodic type on both faces")
        blk = subs[key]
        lo = np.take(blk.x_int, 0, axis=1 + ax)
        hi = np.take(blk.x_int, -1, axis=1 + ax)
        d = (hi - lo).reshape(3, -1)
        if not np.allclose(d, d[:, :1], atol=1e-9 * max(np.abs(blk.x_int).max(), 1.0)):
            raise ValueError(f"{blk.name}: axis {ax} faces are not related by a pure translation")
        mesh.connect(blk, FACE_NAME[(ax, 0)], blk, FACE_NAME[(ax, 1)], shift=tuple(-d[:, 0]))
        done.add((key, ax))

    if finalize:
        mesh.finalize()
    return mesh


def describe(mesh):
    """Human-readable topology summary: blocks, sizes, BC or partner of every face."""
    lines, total = [], 0
    for i, b in enumerate(mesh.blocks):
        total += int(np.prod(b.n))
        o = getattr(b, "origin", None)
        src = f"  (block {o['block'] + 1}, nodes {[tuple(r) for r in o['ranges']]})" if o else ""
        lines.append(f"[{i}] {b.name}: cells {b.n}{src}")
        for (ax, s), f in sorted(b.faces.items()):
            if isinstance(f, Interface):
                nb = mesh.blocks[f.nb]
                sh = "" if not np.any(f.shift) else f"  shift={[float(v) + 0.0 for v in np.round(f.shift, 6)]}"
                lines.append(f"      {FACE_NAME[(ax, s)]}: -> {nb.name}.{FACE_NAME[f.src_face]}{sh}")
            else:
                lines.append(f"      {FACE_NAME[(ax, s)]}: {type(f).__name__}")
    lines.append(f"total cells: {total}")
    return "\n".join(lines)
