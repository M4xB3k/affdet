"""Instance-disjoint split for the detector generalization arms (E5 D-b).

The question this answers: can the detector find a mesh instance it has
never seen, having seen *other* instances of that class? That needs a split
where the held-out meshes appear in NO training scene -- holding out scenes
at random does not give it, because the same mesh recurs across scenes.
"""
import pytest

from affdet.detect.instance_split import (
    held_out_instances,
    scenes_containing,
    split_scenes_by_instance,
)


class _FakeScene:
    def __init__(self, scene_id, obj_ids_classes):
        self.scene_id = scene_id
        self.objects = [type("O", (), {"obj_id": i, "obj_class": c})()
                        for i, c in obj_ids_classes]


def _corpus():
    """3 classes x 5 instances, spread over scenes."""
    scenes = {}
    for cls in ("bowl", "knife", "apple"):
        for i in range(5):
            scenes[f"s_{cls}_{i}"] = _FakeScene(
                f"s_{cls}_{i}", [(f"{cls}_{i:03d}", cls)])
    return scenes


def test_holds_out_a_fraction_of_instances_per_class():
    """Stratified by class: every class contributes held-out meshes, so the
    test set is not accidentally all one category."""
    inst_by_cls = {"bowl": {f"bowl_{i}" for i in range(10)},
                   "knife": {f"knife_{i}" for i in range(10)}}

    held = held_out_instances(inst_by_cls, frac=0.2, seed=0)

    assert len(held) == 4
    assert sum(1 for h in held if h.startswith("bowl")) == 2
    assert sum(1 for h in held if h.startswith("knife")) == 2


def test_holdout_is_deterministic_for_a_seed():
    inst = {"bowl": {f"bowl_{i}" for i in range(10)}}
    assert (held_out_instances(inst, frac=0.3, seed=1)
            == held_out_instances(inst, frac=0.3, seed=1))


def test_different_seeds_choose_different_instances():
    inst = {"bowl": {f"bowl_{i}" for i in range(20)}}
    assert (held_out_instances(inst, frac=0.2, seed=0)
            != held_out_instances(inst, frac=0.2, seed=1))


def test_a_class_too_small_to_split_holds_out_nothing():
    """With 2 instances, holding one out leaves a single training example;
    the class cannot support an instance-generalization claim, and must be
    reported as such rather than silently producing a meaningless number."""
    inst_by_cls = {"kiwifruit": {"kiwifruit_001", "kiwifruit_002"}}

    held = held_out_instances(inst_by_cls, frac=0.2, seed=0, min_train=2)

    assert held == set()


def test_at_least_one_instance_is_held_out_when_the_class_can_afford_it():
    """A 5-instance class at frac=0.2 rounds to 1, not 0 -- rounding a
    class out of the test set entirely would silently drop it."""
    inst_by_cls = {"fork": {f"fork_{i}" for i in range(5)}}

    held = held_out_instances(inst_by_cls, frac=0.2, seed=0)

    assert len(held) == 1


def test_scenes_containing_a_held_out_mesh_are_found():
    corpus = _corpus()
    hits = scenes_containing(corpus, {"bowl_001"})
    assert hits == {"s_bowl_1"}


def test_train_scenes_contain_no_held_out_mesh():
    """The load-bearing property: if a held-out mesh appears anywhere in
    training, the arm measures memorisation, not generalization."""
    corpus = _corpus()
    held = {"bowl_001", "knife_003"}

    train, test = split_scenes_by_instance(corpus, held)

    for sid in train:
        for o in corpus[sid].objects:
            assert o.obj_id not in held


def test_test_scenes_all_contain_a_held_out_mesh():
    corpus = _corpus()
    held = {"bowl_001", "knife_003"}

    train, test = split_scenes_by_instance(corpus, held)

    assert test == {"s_bowl_1", "s_knife_3"}


def test_split_is_a_partition():
    corpus = _corpus()
    train, test = split_scenes_by_instance(corpus, {"bowl_001"})

    assert train | test == set(corpus)
    assert not (train & test)


def test_empty_holdout_puts_everything_in_train():
    corpus = _corpus()
    train, test = split_scenes_by_instance(corpus, set())

    assert test == set()
    assert train == set(corpus)


# ---------------------------------------------------------------------------
# Converter: invisible objects
# ---------------------------------------------------------------------------

def _camera():
    from affdet.schemas import Camera
    return Camera(
        pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        intrinsics=[[50, 0, 32], [0, 50, 32], [0, 0, 1]],
    )

def test_invisible_object_is_skipped_not_fatal(tmp_path):
    """The renderer marks an object invisible when it occupies almost no
    pixels; one such object in the corpus has a zero-width bbox
    (sweet_potato_013 in scene_v3_00888, 2 visible px). It has no detectable
    content, so it is not a training target -- and it must not abort the
    conversion of 12,753 otherwise-good scenes."""
    from PIL import Image

    from affdet.detect.yolo_dataset import convert
    from affdet.schemas import Camera, Scene, SceneObject

    scenes = tmp_path / "scenes"
    scenes.mkdir()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(scenes / "scene_0001.png")
    scene = Scene(
        scene_id="scene_0001", image_path="scene_0001.png",
        image_size=[64, 64], source="blender",
        camera=_camera(),
        objects=[
            SceneObject(obj_id="knife_001", obj_class="knife",
                        bbox=[1, 1, 20, 20], visible=True, visible_px=400),
            SceneObject(obj_id="sweet_potato_013", obj_class="sweet_potato",
                        bbox=[33, 33, 33, 34], visible=False, visible_px=2),
        ],
    )
    (scenes / "scene_0001.json").write_text(scene.model_dump_json())

    data_yaml = convert(scenes, tmp_path / "out", val_fraction=0.0)

    labels = (tmp_path / "out" / "labels" / "train" / "scene_0001.txt")
    assert len(labels.read_text().strip().splitlines()) == 1, \
        "only the visible object becomes a training target"


def test_degenerate_bbox_on_a_VISIBLE_object_still_fails(tmp_path):
    """Skipping invisible objects must not become a blanket excuse: a
    zero-area box on an object the renderer says IS visible is a real defect
    and must still be caught."""
    from PIL import Image

    from affdet.detect.yolo_dataset import convert
    from affdet.schemas import Camera, Scene, SceneObject

    scenes = tmp_path / "scenes"
    scenes.mkdir()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(scenes / "scene_0001.png")
    scene = Scene(
        scene_id="scene_0001", image_path="scene_0001.png",
        image_size=[64, 64], source="blender",
        camera=_camera(),
        objects=[SceneObject(obj_id="knife_001", obj_class="knife",
                             bbox=[10, 10, 10, 20], visible=True,
                             visible_px=500)],
    )
    (scenes / "scene_0001.json").write_text(scene.model_dump_json())

    with pytest.raises(ValueError, match="degenerate"):
        convert(scenes, tmp_path / "out")
