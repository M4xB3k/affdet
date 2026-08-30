"""canCut-only image format: 2 paired-scene angles + 2 solo object tiles in
one 2x2 grid, instead of the standard 4-paired-angle grid used by every
other predicate.

Motivation (2026-07-14, second pass): the type-field prompt fix alone got
box-cutter meshes correctly classified in a live test, but the solo tiles
give a much clearer, unambiguous view of tool TYPE (blade mechanism, handle
material, hinge) than a quartered paired-scene angle does -- adding them is
extra robustness on top of the type-field fix, scoped to canCut only so it
doesn't reintroduce the scale-blindness problem paired renders were built to
fix for size-sensitive predicates like canSupport/canContain.
"""
from pathlib import Path

from PIL import Image

from affdet.label.vlm import compose_cancut_grid


def _make_png(path: Path, color, size=(64, 48)):
    Image.new("RGB", size, color=color).save(path)


def test_compose_cancut_grid_is_2x2_of_the_four_inputs(tmp_path):
    angle0 = tmp_path / "angle_00.png"
    angle1 = tmp_path / "angle_01.png"
    actor_tile = tmp_path / "tile_actor.png"
    recip_tile = tmp_path / "tile_recip.png"
    for p, c in [(angle0, (255, 0, 0)), (angle1, (0, 255, 0)),
                 (actor_tile, (0, 0, 255)), (recip_tile, (255, 255, 0))]:
        _make_png(p, c)

    out_path = tmp_path / "cancut_grid.png"
    compose_cancut_grid(angle0, angle1, actor_tile, recip_tile, out_path)

    img = Image.open(out_path)
    w, h = 64, 48
    assert img.size == (w * 2, h * 2)

    # top-left quadrant should be angle0's color (red), top-right angle1's
    # (green), bottom-left actor tile (blue), bottom-right recipient tile
    # (yellow) -- confirms placement order, not just that *something* was
    # composed.
    assert img.getpixel((10, 10))[:3] == (255, 0, 0)
    assert img.getpixel((w + 10, 10))[:3] == (0, 255, 0)
    assert img.getpixel((10, h + 10))[:3] == (0, 0, 255)
    assert img.getpixel((w + 10, h + 10))[:3] == (255, 255, 0)


def test_compose_cancut_grid_handles_mismatched_tile_sizes(tmp_path):
    # Solo tiles are typically a different resolution than paired-scene
    # angles -- composition must resize instead of crashing.
    angle0 = tmp_path / "angle_00.png"
    angle1 = tmp_path / "angle_01.png"
    actor_tile = tmp_path / "tile_actor.png"
    recip_tile = tmp_path / "tile_recip.png"
    _make_png(angle0, (255, 0, 0), size=(640, 480))
    _make_png(angle1, (0, 255, 0), size=(640, 480))
    _make_png(actor_tile, (0, 0, 255), size=(480, 480))
    _make_png(recip_tile, (255, 255, 0), size=(480, 480))

    out_path = tmp_path / "cancut_grid.png"
    compose_cancut_grid(angle0, angle1, actor_tile, recip_tile, out_path)

    img = Image.open(out_path)
    assert img.size == (640 * 2, 480 * 2)
