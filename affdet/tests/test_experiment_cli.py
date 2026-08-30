from typer.testing import CliRunner
from affdet.scripts.experiment import app
from affdet.experiment.sweep import run_sweep
from affdet.experiment.config import ExperimentConfig
from affdet.render.mock import MockRenderer
import yaml

runner = CliRunner()


def test_cli_help_lists_run():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0 and "run" in r.output


def _labels_for(scenes, tmp_path):
    from affdet.schemas import Scene, Labels, LabelEntry
    lab = tmp_path/"labels"; lab.mkdir()
    for sj in sorted(scenes.glob("scene_*.json")):
        sc = Scene.model_validate_json(sj.read_text())
        e = []
        for a in sc.objects:
            for b in sc.objects:
                if a.obj_id != b.obj_id:
                    e.append(LabelEntry(actor=a.obj_id, recipient=b.obj_id,
                        predicate="canContain", verdict=1, confidence=0.9,
                        source="mock", score=0.9)); break
            break
        (lab/sj.name).write_text(Labels(scene_id=sc.scene_id, tier1=e).model_dump_json())
    return lab


def test_sweep_runs_multiple_configs(tmp_path):
    scenes = tmp_path/"scenes"; MockRenderer(seed=0, n_scenes=8).run(scenes)
    labels = _labels_for(scenes, tmp_path)
    cfgs = []
    for nm in ("a", "b"):
        p = tmp_path/f"{nm}.yaml"
        p.write_text(yaml.safe_dump({"name": nm, "scenes": str(scenes),
                                     "labels": str(labels), "epochs": 1}))
        cfgs.append(p)
    recs = run_sweep(cfgs, tmp_path/"exp", mock=True)
    assert len(recs) == 2 and all("macro_f1" in r["metrics"]["raw"] for r in recs)
