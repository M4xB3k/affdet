from affdet.render.scene_sampler import SceneSampler


def test_scene_config_has_required_fields():
    cfg = SceneSampler(seed=0).sample()
    assert len(cfg.objects) >= 1
    assert cfg.camera_pose.shape == (4, 4)


def test_scene_sampler_deterministic():
    a = SceneSampler(seed=42).sample()
    b = SceneSampler(seed=42).sample()
    assert a.objects == b.objects


def test_sampling_covers_cut_and_contain():
    # YCB sampler covers canCut/canContain/canSupport — peeler/pourables not in YCB meshes
    from affdet.intersection_matrix import is_candidate_pair
    sampler = SceneSampler(seed=0)
    covered = set()
    for _ in range(20):
        cfg = sampler.sample()
        classes = [o.obj_class for o in cfg.objects]
        for a in classes:
            for r in classes:
                if a != r:
                    for pred in ("canCut", "canContain", "canSupport"):
                        if is_candidate_pair(a, r, pred):
                            covered.add(pred)
    assert "canCut" in covered or "canContain" in covered
