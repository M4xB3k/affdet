"""Split Objaverse models that contain several side-by-side objects (e.g. a salt
and pepper shaker pair) into one GLB per object.

Parts (connected mesh components) are grouped when their footprints, seen from
above (glTF: +Y up), overlap by more than 20 % of the smaller one, so a cap stays
with the body under it while objects standing (or touching) side by side
separate. If that gives too few objects, the widest parts (a tray, a holder) are
left out one at a time until the objects come apart.

    python scripts/split_pairs.py IN.glb OUT_DIR [--expect 2] [--preview out.png]
"""
import argparse
import os

import numpy as np
import trimesh


def footprint(m):
    b = m.bounds
    return np.array([b[0, 0], b[0, 2], b[1, 0], b[1, 2]])  # xmin, zmin, xmax, zmax


def overlap_ratio(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 1.0


def group(fps, keep, min_ratio):
    parent = {i: i for i in keep}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in keep:
        for b in keep:
            if a < b and overlap_ratio(fps[a], fps[b]) > min_ratio:
                parent[find(a)] = find(b)
    groups = {}
    for i in keep:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def split(path, expect=2, min_ratio=0.2, max_holders=3):
    scene = trimesh.load(path, force="scene")
    geoms = scene.dump()  # meshes in world frame, one per original material/primitive
    parts, src = [], []  # connected components and (geometry index, face indices) they came from
    for gi, g in enumerate(geoms):
        for faces in trimesh.graph.connected_components(g.face_adjacency, nodes=np.arange(len(g.faces)), min_len=1):
            parts.append(g.submesh([faces], append=True))
            src.append((gi, faces))
    fps = [footprint(p) for p in parts]
    area = [(f[2] - f[0]) * (f[3] - f[1]) for f in fps]
    nverts = sum(len(p.vertices) for p in parts)

    def big_groups(gs):
        # ignore specks (< 2 % of the vertices) when counting objects
        return [g for g in gs if sum(len(parts[i].vertices) for i in g) > 0.02 * nverts]

    # Parts with a large footprint that glue the objects together (tray, holder) are
    # removed one at a time, widest first, until the expected number of objects appears.
    holders = []
    keep = list(range(len(parts)))
    groups = group(fps, keep, min_ratio)
    for i in sorted(keep, key=lambda i: -area[i])[:max_holders]:
        if len(big_groups(groups)) >= expect:
            break
        trial = [k for k in keep if k != i]
        g2 = group(fps, trial, min_ratio)
        if len(big_groups(g2)) > len(big_groups(groups)):
            holders.append(i); keep = trial; groups = g2

    big = big_groups(groups)
    # attach specks to the object whose footprint they overlap most
    for g in groups:
        if g in big:
            continue
        for i in g:
            best = max(big, key=lambda bg: max(overlap_ratio(fps[i], fps[j]) for j in bg))
            best.append(i)
    big.sort(key=lambda g: -sum(area[i] for i in g))

    # cut each original geometry by group, so every object keeps the original parts and materials
    pieces = []
    for g in big:
        per_geom = {}
        for i in g:
            gi, faces = src[i]
            per_geom.setdefault(gi, []).append(faces)
        pieces.append([geoms[gi].submesh([np.concatenate(fs)], append=True) for gi, fs in sorted(per_geom.items())])
    objects = [trimesh.util.concatenate([parts[i] for i in g]) for g in big]
    held_out = [parts[i] for i in holders]
    ok = len(objects) == expect
    if ok:
        vols = [float(np.prod(o.extents)) for o in objects]
        ok = min(vols) / max(vols) > 0.3  # the objects should be of comparable size
    return objects, pieces, held_out, ok


def preview(objects, held_out, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    colors = plt.cm.tab10.colors
    for k, (axis, (u, v), title) in enumerate([(ax[0], (0, 1), "front (x, y)"), (ax[1], (0, 2), "top (x, z)")]):
        for m in held_out:
            axis.scatter(m.vertices[:, u], m.vertices[:, v], s=0.2, c="lightgrey")
        for i, m in enumerate(objects):
            pts = m.vertices if len(m.vertices) < 20000 else m.vertices[np.random.default_rng(0).choice(len(m.vertices), 20000, replace=False)]
            axis.scatter(pts[:, u], pts[:, v], s=0.2, color=colors[i % 10], label=f"object {i}")
        axis.set_aspect("equal")
        axis.set_title(title)
    ax[0].legend(markerscale=20, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("out_dir")
    ap.add_argument("--expect", type=int, default=2)
    ap.add_argument("--preview")
    a = ap.parse_args()
    objects, pieces, held_out, ok = split(a.glb, a.expect)
    uid = os.path.splitext(os.path.basename(a.glb))[0]
    if a.preview:
        preview(objects, held_out, a.preview)
    os.makedirs(a.out_dir, exist_ok=True)
    for i, (m, ps) in enumerate(zip(objects, pieces)):
        # keep each part as its own geometry so every part keeps its own material
        shift = [-m.bounds[:, 0].mean(), -m.bounds[0, 1], -m.bounds[:, 2].mean()]  # base centred at origin
        scene = trimesh.Scene()
        for k, part in enumerate(ps):
            part = part.copy()
            part.apply_translation(shift)
            scene.add_geometry(part, geom_name=f"part_{k}")
        scene.export(os.path.join(a.out_dir, f"{uid}_{i}.glb"))
    print(f"{uid}: {len(objects)} objects, {len(held_out)} holder parts left out, "
          f"extents {[np.round(m.extents, 3).tolist() for m in objects]} -> {'OK' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
