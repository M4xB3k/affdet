"""Load simulation bodies by a short spec: "shaker:d00ef026" (a curated shaker, one convex hull), "mug:0", "bowl:2" ...

Role specs index into the picks JSON (lists of curated objects per role, with objathor height and mass); shaker specs
are the first 8 characters of an object uid in the shaker CSV (target height and mass). Bodies are cached per registry.
"""
from __future__ import annotations

import csv
import json
import os

from affdet.sim.objects import load_body


class Registry:
    def __init__(self, picks_json: str, glb_dir: str, shaker_csv: str, cache_dir: str, friction: float = 0.6):
        self.picks = json.load(open(picks_json))
        self.glb_dir = glb_dir
        self.cache = cache_dir
        self.friction = friction  # friction coefficient of every object (a materials table is not built yet)
        self.shakers = {r["object_uid"][:8]: r for r in csv.DictReader(open(shaker_csv))}
        self._bodies = {}

    def spec_name(self, spec: str) -> str:
        kind, ident = spec.split(":")
        return f"shaker{ident}" if kind == "shaker" else f"{kind}{ident}"

    def get(self, spec: str):
        """(Body, visual mesh) for a spec."""
        if spec not in self._bodies:
            kind, ident = spec.split(":")
            if kind == "shaker":
                r = self.shakers[ident]
                self._bodies[spec] = load_body(os.path.join(self.glb_dir, r["object_uid"] + ".glb"), float(r["target_height_cm"]) / 100,
                                               float(r["target_mass_kg"]), f"shaker_{ident}", up_axis=r.get("up_axis") or "+y",
                                               cache_dir=self.cache, convex=True, friction=self.friction)
            else:
                r = self.picks[kind][int(ident)]
                self._bodies[spec] = load_body(os.path.join(self.glb_dir, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100,
                                               float(r["objathor_mass_kg"]), f"{kind}{ident}", cache_dir=self.cache, friction=self.friction)
        return self._bodies[spec]

    def glb(self, spec: str):
        """(glb path, height in m, up axis): the arguments of `affdet.geometry.prepare.prepare`, for rendering."""
        kind, ident = spec.split(":")
        if kind == "shaker":
            r = self.shakers[ident]
            return os.path.join(self.glb_dir, r["object_uid"] + ".glb"), float(r["target_height_cm"]) / 100, r.get("up_axis") or "+y"
        r = self.picks[kind][int(ident)]
        return os.path.join(self.glb_dir, r["object_uid"] + ".glb"), float(r["objathor_height_cm"]) / 100, "+y"

    def specs(self, kind: str) -> list[str]:
        return [f"{kind}:{i}" for i in range(len(self.picks[kind]))]
