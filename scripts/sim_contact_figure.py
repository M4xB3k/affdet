"""Render a contact-region label pair: A and B painted where they touch at the most robust placement.

    python scripts/sim_contact_figure.py CONTACT_DIR PAIR_NAME OUT_PNG
"""
import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from affdet.sim.label_export import load  # noqa: E402
from scripts.sim_map_render import render  # noqa: E402


def main():
    d, name, out = sys.argv[1:4]
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    for row, side, els in ((0, "A", (35, -35)), (1, "B", (70, 40))):
        arr, meta = load(os.path.join(d, f"{name}.{side}.npz"))
        mz = np.load(os.path.join(d, f"{name}.{side}.mesh.npz"))
        mesh = trimesh.Trimesh(mz["vertices"], mz["faces"], process=False)

        def col(sub):
            c = np.where(arr["contact"][:, None] > 0, np.array([[214, 40, 40, 255]]), np.array([[200, 200, 200, 255]])).astype(np.uint8)
            return c if len(sub.vertices) == len(c) else c[: len(sub.vertices)]
        for k, el in enumerate(els):
            m2 = mesh
            img = render(m2, lambda s: col(s), size=420, el=el, dist=2.0 if side == "A" else 1.6)
            axs[row][k].imshow(img); axs[row][k].axis("off")
            axs[row][k].set_title(f"{'A (put down)' if side == 'A' else 'B (support)'}: red = touching B/A, {'from above' if el > 0 else 'from below'}", fontsize=9)
    fig.suptitle(f"{name}: {meta['face']} face down at ({1000 * meta['x']:.0f}, {1000 * meta['y']:.0f}) mm")
    fig.tight_layout(); fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
