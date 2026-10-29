"""Plot3D preprocessing and METIS partition manifests.

This module deliberately stops at preprocessing.  A partition in the manifest
is ownership metadata for a later multi-device solver; it is not JAX sharding.
"""
from dataclasses import dataclass
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from .airfoil_case import build_case
from .bc import BC
from .mesh import Block, Interface, Mesh, FACE_NAMES


MANIFEST_VERSION = 1
_FACE_NAME = {value: key for key, value in FACE_NAMES.items()}


@dataclass(frozen=True)
class PartitionManifest:
    """Validated METIS assignment and the weights used to produce it."""

    version: int
    partitions: int
    blocks: tuple
    interfaces: tuple
    loads: tuple
    edgecuts: int

    def as_dict(self):
        return {
            "version": self.version,
            "partitions": self.partitions,
            "blocks": list(self.blocks),
            "interfaces": list(self.interfaces),
            "loads": list(self.loads),
            "edgecuts": self.edgecuts,
        }

    def write(self, path):
        with open(path, "w") as stream:
            json.dump(self.as_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")


def _segments(ncells, count):
    if ncells < 3 * count:
        raise ValueError(
            f"cannot split {ncells} cells into {count} pieces with ng=3"
        )
    edges = np.linspace(0, ncells, count + 1, dtype=int)
    if np.min(np.diff(edges)) < 3:
        raise ValueError("split would create a block thinner than ng=3")
    return list(zip(edges[:-1], edges[1:]))


def split_mesh(mesh, target_parts):
    """Split each source block along its longest axis into solver-sized blocks."""
    total = sum(int(np.prod(block.n)) for block in mesh.blocks)
    target_cells = max(1, int(np.ceil(total / target_parts)))
    pieces = []
    source_faces = {}
    cuts = [{axis: set() for axis in range(3)} for _ in mesh.blocks]
    for source, block in enumerate(mesh.blocks):
        axis = int(np.argmax(block.n))
        count = max(1, int(np.ceil(np.prod(block.n) / target_cells)))
        ranges = _segments(block.n[axis], count)
        cuts[source][axis].update(
            boundary for pair in ranges[:-1] for boundary in (pair[1],)
        )

    changed = True
    while changed:
        changed = False
        for source, block in enumerate(mesh.blocks):
            for face, old in block.faces.items():
                if not isinstance(old, Interface):
                    continue
                axis, side = face
                for cut in tuple(cuts[source][tangent] for tangent in range(3)
                                 if tangent != axis):
                    tangent = next(
                        candidate for candidate in range(3)
                        if candidate != axis and cut is cuts[source][candidate]
                    )
                    target_axis = old.mapping[tangent]
                    for boundary in tuple(cut):
                        mapped = (
                            boundary if not old.flip[tangent]
                            else mesh.blocks[old.nb].n[target_axis] - boundary
                        )
                        if mapped not in cuts[old.nb][target_axis]:
                            cuts[old.nb][target_axis].add(mapped)
                            changed = True

    for source, block in enumerate(mesh.blocks):
        axis_ranges = [
            [(0, boundary) for boundary in sorted(cuts[source][axis])]
            for axis in range(3)
        ]
        for axis in range(3):
            boundaries = [0] + sorted(cuts[source][axis]) + [block.n[axis]]
            axis_ranges[axis] = list(zip(boundaries[:-1], boundaries[1:]))
        for piece, ranges in enumerate(product(*axis_ranges)):
            if any(stop - start < mesh.ng for start, stop in ranges):
                raise ValueError("interface split propagation created a block thinner than ng")
            slices = [slice(None)] * 4
            for axis, (start, stop) in enumerate(ranges):
                slices[axis + 1] = slice(start, stop + 1)
            child = Block(
                f"{block.name}_{piece:03d}" if len(axis_ranges) > 1 else block.name,
                block.x_int[tuple(slices)],
                ng=mesh.ng,
            )
            child.origin = {
                "source_block": source,
                "ranges": tuple(
                    ranges[a] for a in range(3)
                ),
            }
            pieces.append(child)
            source_faces[id(child)] = {
                (a, side): block.faces[(a, side)]
                for a in range(3) for side in (0, 1)
                if (side == 0 and ranges[a][0] == 0
                    or side == 1 and ranges[a][1] == block.n[a])
            }

    result = Mesh(pieces)
    by_source = {}
    for child in pieces:
        by_source.setdefault(child.origin["source_block"], []).append(child)

    # Interfaces between the newly created pieces of one source block.
    for source_children in by_source.values():
        for left in source_children:
            for right in source_children:
                if left is right:
                    continue
                for axis in range(3):
                    if left.origin["ranges"][axis][1] == right.origin["ranges"][axis][0]:
                        other = [a for a in range(3) if a != axis]
                        if all(left.origin["ranges"][a] == right.origin["ranges"][a]
                               for a in other):
                            result.connect(left, _FACE_NAME[(axis, 1)],
                                           right, _FACE_NAME[(axis, 0)])

    # Copy external BCs and recover every old interface by coordinate matching.
    pending = []
    for child in pieces:
        for face, old in source_faces[id(child)].items():
            axis, side = face
            face_name = _FACE_NAME[face]
            if isinstance(old, Interface):
                pending.append((child, face_name, old))
            else:
                if not isinstance(old, BC):
                    raise TypeError(f"unsupported boundary object {type(old).__name__}")
                child.set_bc(face_name, old)

    connected = set()
    for child, face_name, old in pending:
        key = (id(child), face_name)
        if key in connected:
            continue
        candidates = [
            (other, _FACE_NAME[(axis, side)])
            for other in by_source[old.nb]
            for axis, side in [(old.src_face[0], old.src_face[1])]
            if (axis, side) in source_faces[id(other)]
        ]
        for other, other_face in candidates:
            try:
                result.connect(child, face_name, other, other_face,
                               shift=old.shift)
            except ValueError:
                continue
            connected.add(key)
            connected.add((id(other), other_face))
            break
        else:
            raise ValueError(
                f"could not match split interface {child.name}.{face_name}"
            )
    result.finalize()
    return result


def _graph(mesh):
    edges = {}
    for index, block in enumerate(mesh.blocks):
        for face, interface in block.faces.items():
            if not isinstance(interface, Interface):
                continue
            other = interface.nb
            if index == other:
                continue
            key = tuple(sorted((index, other)))
            face_cells = int(np.prod([
                block.n[a] for a in range(3) if a != face[0]
            ]))
            edges[key] = max(edges.get(key, 0), face_cells)
    adjacency = [[] for _ in mesh.blocks]
    for (left, right), weight in edges.items():
        adjacency[left].append(right)
        adjacency[left][-1] = (right, weight)
        adjacency[right].append(left)
        adjacency[right][-1] = (left, weight)
    return adjacency, edges


def preprocess_oat15a(grid_path, manifest_path, partitions=8, ng=3):
    """Assemble, split, partition, and manifest an oat15a Plot3D grid."""
    if partitions < 1:
        raise ValueError("partitions must be positive")
    mesh, _, _ = build_case(grid_path)
    if mesh.ng != ng:
        raise ValueError(f"solver mesh uses ng={mesh.ng}, requested ng={ng}")
    mesh = split_mesh(mesh, partitions)
    adjacency, edges = _graph(mesh)
    if len(mesh.blocks) < partitions:
        raise ValueError(
            f"split produced {len(mesh.blocks)} blocks, fewer than {partitions} partitions"
        )

    try:
        import metis
    except ImportError as exc:
        raise RuntimeError(
            "METIS preprocessing requires the metis Python binding in pyvista"
        ) from exc
    edgecuts, assignment = metis.part_graph(
        adjacency, nparts=partitions,
        nodew=[int(np.prod(block.n)) for block in mesh.blocks],
    )
    assignment = [int(value) for value in assignment]
    loads = [0] * partitions
    for block, owner in zip(mesh.blocks, assignment):
        loads[owner] += int(np.prod(block.n))
    if sorted(set(assignment)) != list(range(partitions)):
        raise ValueError(
            f"METIS left partitions empty: assignment={assignment}"
        )

    blocks = tuple({
        "index": index,
        "name": block.name,
        "source_block": block.origin["source_block"],
        "cells": int(np.prod(block.n)),
        "shape": list(map(int, block.n)),
        "partition": owner,
    } for index, (block, owner) in enumerate(zip(mesh.blocks, assignment)))
    interfaces = tuple({
        "blocks": [left, right],
        "face_cells": weight,
        "cross_partition": assignment[left] != assignment[right],
    } for (left, right), weight in sorted(edges.items()))
    manifest = PartitionManifest(
        MANIFEST_VERSION, partitions, blocks, interfaces, tuple(loads), int(edgecuts),
    )
    manifest.write(manifest_path)
    return manifest, {"edgecuts": int(edgecuts), "loads": loads}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path, help="assembled oat15a Plot3D file")
    parser.add_argument("manifest", type=Path, help="JSON manifest to write")
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--ng", type=int, default=3)
    args = parser.parse_args(argv)
    manifest, report = preprocess_oat15a(
        args.grid, args.manifest, args.partitions, args.ng,
    )
    print(
        f"wrote {args.manifest}: {len(manifest.blocks)} blocks, "
        f"{report['edgecuts']} METIS edge cuts, loads={report['loads']}"
    )


if __name__ == "__main__":
    main()
