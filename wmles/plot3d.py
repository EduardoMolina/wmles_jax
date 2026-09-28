"""Plot3D multiblock grid I/O (whole-format, 3-D, structured).

read_plot3d(path)  -> Plot3DGrid(nodes=[(3,ni,nj,nk) float64, ...], iblank=[...] or None)
write_plot3d(path, nodes, formatted=True, ...)

Supported
* formatted (ASCII) and unformatted (Fortran sequential records) files, auto-detected
* unformatted: little/big endian and 4/8-byte reals auto-detected from the record markers
* optional iblank array (auto-detected from the value count in formatted files; from the
  record length in unformatted files)
* 3-D multiblock ("MG") files, the layout used by the NASA TMBWG grids. A "2-D" file
  (2 coordinate arrays per block) is read too and returned with a singleton k axis and
  z = 0; use `wmles.nmf.extrude` to turn it into a 3-D block.

Node arrays are (3, ni, nj, nk) with x,y,z stacked on axis 0, i fastest in the file.
"""
from dataclasses import dataclass
import struct

import numpy as np


@dataclass
class Plot3DGrid:
    nodes: list          # list of (3, ni, nj, nk) float64
    iblank: object       # None or list of (ni, nj, nk) int arrays
    formatted: bool
    ndim: int = 3


def _is_ascii(raw):
    head = raw[:200]
    try:
        head.decode("ascii")
    except UnicodeDecodeError:
        return False
    return all(chr(c).isprintable() or chr(c) in "\r\n\t" for c in head)


def _split_dims(vals_int, nb, ndim):
    dims = [tuple(vals_int[ndim * b: ndim * (b + 1)]) for b in range(nb)]
    return [d + (1,) * (3 - ndim) for d in dims]


def _blocks_from_flat(vals, dims, ndim, iblank):
    """vals: 1-D float array of all coordinate (and iblank) values, block by block."""
    nodes, ibl, off = [], [], 0
    for d in dims:
        n = int(np.prod(d))
        arr = vals[off: off + ndim * n].reshape(ndim, n)
        off += ndim * n
        X = np.zeros((3,) + d)
        for c in range(ndim):
            X[c] = arr[c].reshape(d, order="F")
        nodes.append(X)
        if iblank:
            ibl.append(vals[off: off + n].astype(np.int64).reshape(d, order="F"))
            off += n
    assert off == len(vals), "trailing values in Plot3D file"
    return nodes, (ibl if iblank else None)


def _read_formatted(text, ndim):
    toks = text.replace("D", "E").replace("d", "e").split()
    nb = int(toks[0])
    if ndim is None:                        # decide 2-D vs 3-D from the value count
        for nd in (3, 2):
            try:
                dims_try = _split_dims([int(t) for t in toks[1: 1 + nb * nd]], nb, nd)
            except ValueError:
                continue
            npts = sum(int(np.prod(d)) for d in dims_try)
            rest = len(toks) - 1 - nb * nd
            if rest in (nd * npts, (nd + 1) * npts):
                ndim = nd
                break
        else:
            raise ValueError("cannot decide whether this Plot3D file is 2-D or 3-D")
    dims = _split_dims([int(t) for t in toks[1: 1 + nb * ndim]], nb, ndim)
    npts = sum(int(np.prod(d)) for d in dims)
    vals = np.array(toks[1 + nb * ndim:], dtype=float)
    if len(vals) == ndim * npts:
        iblank = False
    elif len(vals) == (ndim + 1) * npts:
        iblank = True
    else:
        raise ValueError(f"value count {len(vals)} matches neither {ndim * npts} (no iblank) "
                         f"nor {(ndim + 1) * npts} (iblank)")
    nodes, ibl = _blocks_from_flat(vals, dims, ndim, iblank)
    return Plot3DGrid(nodes, ibl, True, ndim)


def _read_records(raw):
    """Parse Fortran sequential records, detecting endianness. Returns list of bytes."""
    for endian in ("<", ">"):
        recs, pos, ok = [], 0, True
        while pos < len(raw):
            if pos + 4 > len(raw):
                ok = False
                break
            (n,) = struct.unpack(endian + "i", raw[pos: pos + 4])
            if n < 0 or pos + 8 + n > len(raw):
                ok = False
                break
            (n2,) = struct.unpack(endian + "i", raw[pos + 4 + n: pos + 8 + n])
            if n2 != n:
                ok = False
                break
            recs.append(raw[pos + 4: pos + 4 + n])
            pos += 8 + n
        if ok and recs:
            return recs, endian
    raise ValueError("not a valid Fortran-unformatted Plot3D file")


def _read_unformatted(raw, ndim):
    recs, e = _read_records(raw)
    nb = struct.unpack(e + "i", recs[0])[0]
    dimrec = recs[1]
    nint = len(dimrec) // 4
    if ndim is None:
        if nint == 3 * nb:
            ndim = 3
        elif nint == 2 * nb:
            ndim = 2
        else:
            raise ValueError("dimension record length matches neither 2-D nor 3-D")
    dims = _split_dims(list(struct.unpack(e + f"{nint}i", dimrec)), nb, ndim)
    body = recs[2:]
    if len(body) != nb:
        raise ValueError(f"expected {nb} coordinate records, found {len(body)} "
                         f"(only whole-format multiblock files are supported)")
    nodes, ibl, any_ibl = [], [], False
    for d, rec in zip(dims, body):
        n = int(np.prod(d))
        for rsize, dt in ((4, "f4"), (8, "f8")):
            if len(rec) == ndim * n * rsize:
                vals, has_ibl = np.frombuffer(rec, dtype=e + dt).astype(float), False
                break
            if len(rec) == ndim * n * rsize + n * 4:          # iblank stored as int32
                cut = ndim * n * rsize
                vals = np.concatenate([np.frombuffer(rec[:cut], dtype=e + dt).astype(float),
                                       np.frombuffer(rec[cut:], dtype=e + "i4").astype(float)])
                has_ibl = True
                break
        else:
            raise ValueError(f"record of {len(rec)} bytes doesn't fit dims {d}")
        nn, ii = _blocks_from_flat(vals, [d], ndim, has_ibl)
        nodes += nn
        any_ibl |= has_ibl
        if has_ibl:
            ibl += ii
    return Plot3DGrid(nodes, ibl if any_ibl else None, False, ndim)


def read_plot3d(path, formatted=None, ndim=None):
    """Read a whole-format multiblock Plot3D grid. `formatted`/`ndim` are auto-detected
    when None. Use double precision files where possible (the TMBWG grids need it)."""
    with open(path, "rb") as f:
        raw = f.read()
    if formatted is None:
        formatted = _is_ascii(raw)
    if formatted:
        return _read_formatted(raw.decode("ascii"), ndim)
    return _read_unformatted(raw, ndim)


def write_plot3d(path, nodes, formatted=True, iblank=None, dtype="f8", endian="<"):
    """Write (3, ni, nj, nk) node arrays as a whole-format 3-D multiblock Plot3D file."""
    nodes = [np.asarray(X, float) for X in nodes]
    if formatted:
        with open(path, "w") as f:
            f.write(f"{len(nodes):12d}\n")
            for X in nodes:
                f.write("".join(f"{s:12d}" for s in X.shape[1:]) + "\n")
            for b, X in enumerate(nodes):
                flat = np.concatenate([X[c].ravel(order="F") for c in range(3)])
                if iblank is not None:
                    flat = np.concatenate([flat, np.asarray(iblank[b]).ravel(order="F")])
                for k in range(0, len(flat), 3):
                    f.write("".join(f"{v:26.16E}" for v in flat[k: k + 3]) + "\n")
        return

    def rec(payload):
        return struct.pack(endian + "i", len(payload)) + payload + struct.pack(endian + "i", len(payload))

    with open(path, "wb") as f:
        f.write(rec(struct.pack(endian + "i", len(nodes))))
        dims = [i for X in nodes for i in X.shape[1:]]
        f.write(rec(struct.pack(endian + f"{len(dims)}i", *dims)))
        for b, X in enumerate(nodes):
            payload = b"".join(X[c].ravel(order="F").astype(endian + dtype).tobytes() for c in range(3))
            if iblank is not None:
                payload += np.asarray(iblank[b]).ravel(order="F").astype(endian + "i4").tobytes()
            f.write(rec(payload))
