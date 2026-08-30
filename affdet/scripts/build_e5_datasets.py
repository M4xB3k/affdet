"""Build the two E5 detector-training datasets from the CURRENT corpus.

Two arms, mirroring the affordance model's generalization ladder:

  D-a  scene-disjoint    — held-out arrangements of meshes seen in training.
                           Answers "unseen arrangement?"
  D-b  instance-disjoint — every scene containing a reserved mesh is pulled
                           out of training, so the detector meets meshes it
                           has never seen, having seen others of that class.
                           Answers "unseen object instance, seen class?"

Both are built from the same corpus with the same pinned class vocabulary,
so the two arms' numbers are directly comparable and class ids never shift
between them.

    python -m affdet.scripts.build_e5_datasets build \\
        --scenes /scratch/$USER/omni_scenes_all_merged \\
        --out-root /scratch/$USER/affdet/e5_datasets
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import typer

from affdet.detect.instance_split import (
    held_out_instances,
    split_scenes_by_instance,
)
from affdet.detect.yolo_boxes import _find_scene_jsons
from affdet.detect.yolo_dataset import convert
from affdet.schemas import Scene

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def build(
    scenes: Path = typer.Option(..., help="Current render corpus"),
    out_root: Path = typer.Option(..., help="Where the datasets are written"),
    holdout_frac: float = typer.Option(
        0.05,
        help="Fraction of mesh instances per class to reserve. Scenes average "
             "~3.8 objects, so the strict rule (any reserved mesh -> test) "
             "amplifies this: 0.05 already sends ~30%% of scenes to test, "
             "0.20 sends ~59%% and starves training."),
    seed: int = typer.Option(0),
    val_fraction: float = typer.Option(0.1),
) -> None:
    scene_jsons = _find_scene_jsons(scenes)
    typer.echo(f"[e5-data] {len(scene_jsons)} scenes")

    parsed: dict[str, Scene] = {}
    inst_by_cls: dict[str, set[str]] = collections.defaultdict(set)
    for p in scene_jsons:
        s = Scene.model_validate_json(p.read_text())
        parsed[p.stem] = s
        for o in s.objects:
            inst_by_cls[o.obj_class].add(o.obj_id)

    # Pin the vocabulary to the WHOLE corpus. Deriving it per-arm would let
    # an excluded scene shrink the class list and shift every class id, so
    # the two arms would not be comparable.
    all_classes = set(inst_by_cls)
    typer.echo(f"[e5-data] {len(all_classes)} classes, "
               f"{sum(len(v) for v in inst_by_cls.values())} mesh instances")

    held = held_out_instances(inst_by_cls, frac=holdout_frac, seed=seed)
    train_ids, test_ids = split_scenes_by_instance(parsed, held)
    covered = {h.rsplit("_", 1)[0] for h in held}
    untestable = sorted(all_classes - covered)

    typer.echo(f"[e5-data] instance holdout: {len(held)} meshes reserved, "
               f"train={len(train_ids)} test={len(test_ids)} scenes")
    typer.echo(f"[e5-data] classes contributing novel meshes: "
               f"{len(covered)}/{len(all_classes)}")
    if untestable:
        typer.echo(f"[e5-data] UNTESTABLE at this frac (too few instances to "
                   f"leave a trainable remainder): {untestable}")

    out_root.mkdir(parents=True, exist_ok=True)

    # D-a: scene-disjoint. The converter's own seeded val split is exactly
    # this -- held-out arrangements, meshes seen in training.
    scene_dir = out_root / "scene_disjoint"
    typer.echo(f"[e5-data] building D-a (scene-disjoint) -> {scene_dir}")
    da = convert(scenes, scene_dir, val_fraction=val_fraction, seed=seed,
                 allowed_classes=all_classes)

    # D-b: instance-disjoint. Train on scenes free of every reserved mesh;
    # the val split here IS the novel-mesh test set, so Ultralytics reports
    # generalization directly.
    inst_dir = out_root / "instance_disjoint"
    typer.echo(f"[e5-data] building D-b (instance-disjoint) -> {inst_dir}")
    db = _convert_with_explicit_split(
        scenes, inst_dir, train_ids, test_ids, all_classes)

    manifest = {
        "corpus": str(scenes),
        "seed": seed,
        "holdout_frac": holdout_frac,
        "n_scenes": len(scene_jsons),
        "n_classes": len(all_classes),
        "classes": sorted(all_classes),
        "held_out_instances": sorted(held),
        "n_train_scenes": len(train_ids),
        "n_test_scenes": len(test_ids),
        "untestable_classes": untestable,
        "instances_per_class": {c: len(v) for c, v in sorted(inst_by_cls.items())},
        "data_yaml": {"scene_disjoint": str(da), "instance_disjoint": str(db)},
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    typer.echo(f"[e5-data] wrote {out_root / 'manifest.json'}")


@app.command()
def show(
    out_root: Path = typer.Option(..., help="Dataset root to summarise"),
) -> None:
    """Print a built manifest — split sizes and untestable classes."""
    m = json.loads((out_root / "manifest.json").read_text())
    typer.echo(f"corpus={m['corpus']}")
    typer.echo(f"scenes={m['n_scenes']} classes={m['n_classes']} "
               f"seed={m['seed']} holdout_frac={m['holdout_frac']}")
    typer.echo(f"held-out meshes={len(m['held_out_instances'])} "
               f"train={m['n_train_scenes']} test={m['n_test_scenes']}")
    if m["untestable_classes"]:
        typer.echo(f"untestable classes: {m['untestable_classes']}")


def _convert_with_explicit_split(
    scenes: Path,
    out_dir: Path,
    train_ids: set[str],
    test_ids: set[str],
    all_classes: set[str],
) -> Path:
    """Convert with a caller-chosen train/val assignment.

    ``convert`` splits randomly; here the assignment IS the experiment, so
    it is built directly: train gets scenes free of every reserved mesh,
    val gets the scenes containing them.
    """
    import yaml as _yaml

    from affdet.detect.yolo_dataset import _label_lines

    names = sorted(all_classes)
    class_ids = {n: i for i, n in enumerate(names)}

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for p in _find_scene_jsons(scenes):
        split = "train" if p.stem in train_ids else (
            "val" if p.stem in test_ids else None)
        if split is None:
            continue
        scene = Scene.model_validate_json(p.read_text())
        image_path = (p.parent / scene.image_path).resolve()
        link = out_dir / "images" / split / image_path.name
        if not link.is_symlink():
            link.symlink_to(image_path)
        (out_dir / "labels" / split / f"{p.stem}.txt").write_text(
            "\n".join(_label_lines(scene, class_ids)) + "\n")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(_yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }, sort_keys=False))
    return data_yaml


if __name__ == "__main__":
    app()
