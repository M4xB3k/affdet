import numpy as np, torch
from pathlib import Path
from affdet.schemas import Scene, SceneObject, Camera, Labels, LabelEntry, Features
from affdet.model.train import Trainer
from affdet.model.infer import Inferencer


def _mini(tmp_path):
    # NOTE: writes one scene per call (task-4 brief's literal _mini()), used
    # as-is by test_inferencer_uses_injected_factory below.
    sc = tmp_path/"s"; sc.mkdir(); ft = tmp_path/"f"; ft.mkdir(); lb = tmp_path/"l"; lb.mkdir()
    objs=[SceneObject(obj_id="knife_1",obj_class="knife",bbox=[0,0,10,10]),
          SceneObject(obj_id="apple_1",obj_class="apple",bbox=[0,0,10,10])]
    s=Scene(scene_id="scene_0001",image_path="x.png",image_size=[128,96],source="blender",
            objects=objs,camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                       intrinsics=[[1,0,0],[0,1,0],[0,0,1]]))
    (sc/"scene_0001.json").write_text(s.model_dump_json())
    Features(obj_ids=["knife_1","apple_1"],features=np.zeros((2,768),dtype=np.float32)).save(ft/"scene_0001.npz")
    (lb/"scene_0001.json").write_text(Labels(scene_id="scene_0001",tier1=[
        LabelEntry(actor="knife_1",recipient="apple_1",predicate="canCut",verdict=1,confidence=0.9,source="t",score=0.8)]).model_dump_json())
    return sc,ft,lb


def _mini_multi(tmp_path, n_scenes, feat_dim=768):
    # NOTE: deviates from the task-4 brief's literal single-scene _mini() to
    # work around a pre-existing edge case in affdet/model/dataset.py (out of
    # scope for this task): AffordanceDataset splits scenes by
    # int(0.70*n)/int(0.15*n) with no minimum-1 guard, so a single scene
    # yields 0 train AND 0 val pairs, which crashes
    # DataLoader(..., shuffle=True) (RandomSampler needs num_samples > 0).
    # Every pre-existing AffordanceDataset test sidesteps this by passing
    # split=None; Trainer.fit hardcodes split="train"/"val", so this is the
    # first test to exercise that path end-to-end. n_scenes=7 is the minimum
    # that gives both a non-empty train split (int(0.70*7)=4) and a
    # non-empty val split (int(0.15*7)=1), so Trainer.fit's val loop actually
    # runs and writes best.pt — needed for test_trainer_uses_injected_factory
    # to assert on it. See task-4-report.md for full detail.
    sc = tmp_path/"s"; sc.mkdir(); ft = tmp_path/"f"; ft.mkdir(); lb = tmp_path/"l"; lb.mkdir()
    for n in range(1, n_scenes + 1):
        objs=[SceneObject(obj_id=f"knife_{n}",obj_class="knife",bbox=[0,0,10,10]),
              SceneObject(obj_id=f"apple_{n}",obj_class="apple",bbox=[0,0,10,10])]
        sid = f"scene_{n:04d}"
        s=Scene(scene_id=sid,image_path="x.png",image_size=[128,96],source="blender",
                objects=objs,camera=Camera(pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                           intrinsics=[[1,0,0],[0,1,0],[0,0,1]]))
        (sc/f"{sid}.json").write_text(s.model_dump_json())
        Features(obj_ids=[f"knife_{n}",f"apple_{n}"],features=np.zeros((2,feat_dim),dtype=np.float32)).save(ft/f"{sid}.npz")
        (lb/f"{sid}.json").write_text(Labels(scene_id=sid,tier1=[
            LabelEntry(actor=f"knife_{n}",recipient=f"apple_{n}",predicate="canCut",verdict=1,confidence=0.9,source="t",score=0.8)]).model_dump_json())
    return sc,ft,lb


def test_trainer_unbounded_trains_and_evaluates_end_to_end(tmp_path):
    """Trainer(unbounded=True) must fit without crashing and still write a
    checkpoint — the val-F1 computation switched from dataset._masks to
    dataset._candidate_masks (see affdet/model/train.py), so this guards the
    _active_idx-order assumption that swap depends on."""
    sc, ft, lb = _mini_multi(tmp_path, 7); out = tmp_path/"ck"
    Trainer(epochs=1, unbounded=True).fit(sc, ft, lb, out)
    assert (out/"best.pt").exists()


def test_trainer_unbounded_neg_ratio_trains_end_to_end(tmp_path):
    sc, ft, lb = _mini_multi(tmp_path, 7); out = tmp_path/"ck"
    Trainer(epochs=1, unbounded=True, unbounded_neg_ratio=2).fit(sc, ft, lb, out)
    assert (out/"best.pt").exists()


def test_trainer_uses_injected_factory(tmp_path):
    sc,ft,lb=_mini_multi(tmp_path, 7); out=tmp_path/"ck"
    called={"n":0}
    def factory(input_dim):
        called["n"]+=1
        from affdet.model.mlp import AffordanceMLP
        return AffordanceMLP(input_dim=input_dim)
    Trainer(epochs=1, model_factory=factory).fit(sc,ft,lb,out)
    assert called["n"] >= 1
    assert (out/"best.pt").exists()


def test_inferencer_uses_injected_factory(tmp_path):
    sc,ft,lb=_mini(tmp_path); ck=tmp_path/"ck"; ck.mkdir()
    from affdet.model.mlp import AffordanceMLP
    torch.save({"model_state": AffordanceMLP().state_dict()}, ck/"best.pt")
    called={"n":0}
    def factory(input_dim):
        called["n"]+=1; return AffordanceMLP(input_dim=input_dim)
    Inferencer(model_factory=factory).predict(sc,ft,lb,ck,tmp_path/"pred")
    assert called["n"] >= 1


def test_factory_receives_true_feature_dim(tmp_path):
    """Backbone knob: with vits14-width features (384/obj => x is 384*2+8=776),
    the factory must be handed 776, not the hardcoded vitb14 1544. This is the
    end-to-end path a `backbone: vits14` config exercises."""
    sc,ft,lb=_mini_multi(tmp_path, 7, feat_dim=384); out=tmp_path/"ck"
    seen={"dim":None}
    def factory(input_dim):
        seen["dim"]=input_dim
        from affdet.model.mlp import AffordanceMLP
        return AffordanceMLP(input_dim=input_dim)
    Trainer(epochs=1, model_factory=factory).fit(sc,ft,lb,out)
    assert seen["dim"] == 384*2 + 8, f"expected 776, got {seen['dim']}"
    assert (out/"best.pt").exists()

    # And the Inferencer must reconstruct the same width from the checkpoint,
    # not assume 1544 — otherwise load_state_dict fails on shape mismatch.
    seen_inf={"dim":None}
    def inf_factory(input_dim):
        seen_inf["dim"]=input_dim
        from affdet.model.mlp import AffordanceMLP
        return AffordanceMLP(input_dim=input_dim)
    inf_root=tmp_path/"inf"; inf_root.mkdir()
    sc1,ft1,lb1=_mini_multi(inf_root, 7, feat_dim=384)
    Inferencer(model_factory=inf_factory).predict(sc1,ft1,lb1,out,tmp_path/"pred")
    assert seen_inf["dim"] == 776, f"inferencer expected 776, got {seen_inf['dim']}"
