import json
from pathlib import Path

from affdet.eval.baseline_vlm_runner import run_baseline_eval


class _FakeClient:
    def __init__(self, canned_response: str):
        self._canned = canned_response
        self.calls = []

    class _Chat:
        def __init__(self, outer):
            self._outer = outer

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.calls.append(kwargs)

                class _Msg:
                    content = self._outer._canned

                class _Choice:
                    message = _Msg()

                class _Usage:
                    prompt_tokens = 100
                    completion_tokens = 50
                    completion_tokens_details = None

                class _Resp:
                    choices = [_Choice()]
                    usage = _Usage()

                return _Resp()

        def __getattr__(self, name):
            if name == "completions":
                return self._Completions(self._outer)
            raise AttributeError(name)

    def __getattr__(self, name):
        if name == "chat":
            return self._Chat(self)
        raise AttributeError(name)


def test_run_baseline_eval_calls_client_once_per_photo(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    gt = {
        "image": "DSC00001.JPG",
        "objects": ["knife_1", "apple_1"],
        "positive_pairs": [
            {"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut"}
        ],
    }
    (gt_dir / "gt2_DSC00001.json").write_text(json.dumps(gt))
    (photos_dir / "DSC00001.JPG").write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")

    canned = json.dumps({
        "objects": [{"id": "k1", "category": "knife"}, {"id": "a1", "category": "apple"}],
        "affordances": [
            {"actor": "k1", "recipient": "a1", "predicate": "canCut", "verdict": 1},
        ],
    })
    client = _FakeClient(canned)

    results = run_baseline_eval(gt_dir=gt_dir, photos_dir=photos_dir,
                                 client=client, model="gpt-4o")

    assert len(client.calls) == 1
    assert results["per_photo"]["DSC00001"]["tp"] == 1
    assert "elapsed_seconds" in results["per_photo"]["DSC00001"]
    assert results["totals"]["tp"] == 1


def test_run_baseline_eval_processes_every_gt_file(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    for name in ("DSC00001", "DSC00002"):
        gt = {"image": f"{name}.JPG", "objects": ["knife_1", "apple_1"],
              "positive_pairs": []}
        (gt_dir / f"gt2_{name}.json").write_text(json.dumps(gt))
        (photos_dir / f"{name}.JPG").write_bytes(b"\xff\xd8\xff\xe0fake")

    canned = json.dumps({"objects": [], "affordances": []})
    client = _FakeClient(canned)

    results = run_baseline_eval(gt_dir=gt_dir, photos_dir=photos_dir,
                                 client=client, model="gpt-4o")

    assert len(client.calls) == 2
    assert set(results["per_photo"].keys()) == {"DSC00001", "DSC00002"}
