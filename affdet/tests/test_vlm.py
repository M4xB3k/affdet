import json
import time
from pathlib import Path

import pytest

from affdet.label.vlm import OmniLabeler, build_pairs
from affdet.label.prompts import build_user_prompt, build_system_prompt, AFFORDANCE_QUESTION
from affdet.label.rate_limiter import TokenRateLimiter


def _write_manifest(renders_dir: Path, actor: str, recipient: str,
                     container_flipped: bool = False, with_grid: bool = True) -> Path:
    pair_dir = renders_dir / f"{actor}__{recipient}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "actor": actor, "recipient": recipient,
        "grid": {"rows": 2, "cols": 2}, "n_angles": 4,
        "resolution": [640, 480],
        "angle_files": ["angle_00.png", "angle_01.png", "angle_02.png", "angle_03.png"],
        "grid_file": "grid.png",
        "container_flipped": container_flipped,
    }
    (pair_dir / "manifest.json").write_text(json.dumps(manifest))
    if with_grid:
        (pair_dir / "grid.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return pair_dir


def test_build_pairs_gates_by_intersection_matrix(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_011", "recipient": "durian_030"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_011", "durian_030")

    jobs = build_pairs(pairs_json, renders_dir)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["actor_cat"] == "dish"
    assert job["recipient_cat"] == "durian"
    # dish is a container -> both canContain and canSupport are candidates;
    # dish is not a knife/fork/peeler/pourable, so no other predicates.
    assert set(job["predicates"]) == {"canContain", "canSupport"}


def test_build_pairs_fork_only_pair_gets_pierce_job(tmp_path):
    # 2026-08-03: canPierce moved from a hardcoded rule to a real VLM
    # judgment -- a blunt-tip novelty knife can't actually pierce anything,
    # which the old "true for any knife/fork" rule missed entirely. A
    # fork/apple pair now surfaces canPierce as a real VLM job.
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "fork_002", "recipient": "apple_016"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "fork_002", "apple_016")

    jobs = build_pairs(pairs_json, renders_dir)
    assert len(jobs) == 1
    assert jobs[0]["predicates"] == ["canPierce"]


def test_build_pairs_excludes_canpourinto_entirely(tmp_path):
    # teapot -> bowl is a structurally valid canPourInto candidate per
    # is_candidate_pair(), but canPourInto's truth comes from a rule table
    # (resolved_instances.py's pourable_from/pourable_into), never a VLM
    # judgment -- a teapot/bowl pair should never reach the VLM at all.
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "teapot_022", "recipient": "bowl_003"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "teapot_022", "bowl_003")

    jobs = build_pairs(pairs_json, renders_dir)
    assert jobs == []


def test_build_pairs_knife_gets_both_cut_and_pierce(tmp_path):
    # 2026-08-03: canPierce is now a real VLM judgment (was hardcoded true
    # for any knife/fork) -- a knife/ginger pair surfaces BOTH canCut and
    # canPierce as VLM jobs, since a knife's tip sharpness (piercing) is
    # judged independently of its blade type (cutting).
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "knife_013", "recipient": "ginger_026"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "knife_013", "ginger_026")

    jobs = build_pairs(pairs_json, renders_dir)
    assert len(jobs) == 1
    assert set(jobs[0]["predicates"]) == {"canCut", "canPierce"}


def test_build_pairs_includes_pair_dir_for_cancut_grid_composition(tmp_path):
    # _label_pair needs the pair's directory (not just grid_path) to find
    # angle_00.png/angle_01.png when composing the canCut-only 2x2 grid
    # (2 paired angles + 2 solo tiles).
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "knife_013", "recipient": "ginger_026"}]))
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "knife_013", "ginger_026")

    jobs = build_pairs(pairs_json, renders_dir)
    assert jobs[0]["pair_dir"] == pair_dir


def test_build_pairs_skips_structurally_impossible_pair(tmp_path):
    # knife -> knife: same class, never a candidate for anything
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "knife_001", "recipient": "knife_002"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "knife_001", "knife_002")

    jobs = build_pairs(pairs_json, renders_dir)
    assert jobs == []


def test_build_pairs_skips_missing_manifest(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_001", "recipient": "apple_002"}]))
    renders_dir = tmp_path / "renders"  # no manifest written at all

    jobs = build_pairs(pairs_json, renders_dir)
    assert jobs == []


def test_build_pairs_flags_flipped_container_as_auto_negative(tmp_path):
    # A flipped container can't be judged from the image, but it genuinely
    # can't contain/support in that pose either -- still gets a job (no gap
    # in the corpus), just flagged so run() writes a rule-based negative
    # instead of spending an API call on a broken image.
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_004", "recipient": "broccoli_015"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_004", "broccoli_015", container_flipped=True)

    jobs = build_pairs(pairs_json, renders_dir)
    assert len(jobs) == 1
    assert jobs[0]["auto_negative"] is True
    assert set(jobs[0]["predicates"]) <= {"canContain", "canSupport"}


def test_build_pairs_flipped_container_without_grid_still_gets_job(tmp_path):
    # Auto-negative jobs don't need a usable grid image at all -- the pair
    # still needs a label, just not a visual one.
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_004", "recipient": "broccoli_015"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_004", "broccoli_015",
                     container_flipped=True, with_grid=False)

    jobs = build_pairs(pairs_json, renders_dir)
    assert len(jobs) == 1
    assert jobs[0]["auto_negative"] is True


def test_build_pairs_skips_missing_grid_image(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_001", "recipient": "apple_002"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_001", "apple_002", with_grid=False)

    jobs = build_pairs(pairs_json, renders_dir)
    assert jobs == []


def test_build_user_prompt_names_objects_without_instance_suffix():
    prompt = build_user_prompt("dish", "durian", ["canContain", "canSupport"], 2, 2)
    assert "dish" in prompt
    assert "durian" in prompt
    assert "_0" not in prompt  # no leaked instance id suffix like dish_011
    assert "canContain" in prompt and "canSupport" in prompt
    assert "canCut" not in prompt  # only requested predicates appear


def test_build_user_prompt_mentions_grid_shape():
    prompt = build_user_prompt("bowl", "apple", ["canContain"], 3, 2)
    assert "3x2" in prompt
    assert "6 different camera angles" in prompt


def test_system_prompt_has_no_tile_scale_workaround_language():
    # The old tile-based prompt told the model to distrust the image and
    # assume "kitchen proportions" instead -- that instruction is wrong now
    # that real scale is visible in the paired render, and must not survive.
    prompt = build_system_prompt(["canContain", "canSupport"])
    assert "SIZE ASSUMPTION" not in prompt
    assert "assume real-world kitchen proportions" not in prompt.lower()


def test_system_prompt_asks_for_score_only_no_reasoning():
    # Output must be score-only -- no rationale text -- to keep visible
    # (and GPT-5-mini's reasoning-effort-driven hidden) tokens minimal.
    # canSupport is the one exception (see test below): it additionally asks
    # for a short size_note, since making the model articulate its size
    # comparison measurably stabilizes the judgment (validated 2026-07-14).
    prompt = build_system_prompt(["canContain"])
    assert "score" in prompt
    assert "no reasoning" in prompt.lower()


def test_system_prompt_cansupport_requests_size_note_alongside_score():
    prompt = build_system_prompt(["canSupport"])
    assert "size_note" in prompt


def test_system_prompt_only_includes_requested_calibration():
    # A dish/durian pair only ever needs canContain + canSupport -- it should
    # never see canCut/canPierce/canPeel calibration text, which is both
    # wasted tokens and irrelevant noise for a pair that can never be asked
    # those questions.
    prompt = build_system_prompt(["canContain", "canSupport"])
    assert "canContain calibration" in prompt
    assert "canSupport calibration" in prompt
    assert "canCut calibration" not in prompt
    assert "canPierce calibration" not in prompt
    assert "canPeel calibration" not in prompt
    assert "canPourInto calibration" not in prompt


def test_system_prompt_includes_only_the_one_requested_predicate():
    prompt = build_system_prompt(["canCut"])
    assert "canCut calibration" in prompt
    for other in ("canPierce", "canPeel", "canContain", "canSupport", "canPourInto"):
        assert f"{other} calibration" not in prompt


class _FakeClient:
    def __init__(self, response_json):
        self.response_json = response_json
        self.calls = []

        class _Chat:
            def __init__(self, outer):
                self.completions = self

            def create(inner_self, **kwargs):
                self.calls.append(kwargs)

                class _Msg:
                    content = json.dumps(self.response_json)

                class _Choice:
                    message = _Msg()
                    finish_reason = "stop"

                class _TokDetails:
                    reasoning_tokens = 0

                class _Usage:
                    prompt_tokens = 10
                    completion_tokens = 5
                    completion_tokens_details = _TokDetails()

                class _Resp:
                    choices = [_Choice()]
                    usage = _Usage()

                return _Resp()

        self.chat = _Chat(self)


def test_label_pair_normalizes_cansupport_dict_response_to_plain_score(tmp_path):
    # canSupport's prompt asks for {"score": N, "size_note": "..."} to make
    # the model articulate its size comparison (validated 2026-07-14: this
    # measurably stabilizes the judgment vs. a bare integer), but downstream
    # code (omni_score_labels.py, Labels schema) only ever wants a plain int
    # score -- _label_pair must unwrap the dict, discarding size_note.
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "dish_011", "durian_030")

    job = {
        "actor_inst": "dish_011", "recipient_inst": "durian_030",
        "actor_cat": "dish", "recipient_cat": "durian",
        "predicates": ["canContain", "canSupport"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }
    client = _FakeClient({
        "canContain": 7,
        "canSupport": {"score": 2, "size_note": "durian much larger than dish"},
    })
    labeler = OmniLabeler(client=client)
    result = labeler._label_pair(client, job, job["predicates"])

    assert result["canContain"] == 7
    assert result["canSupport"] == 2  # unwrapped, not the dict


def test_label_pair_uses_composed_grid_for_cancut_when_tiles_available(tmp_path):
    from PIL import Image

    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "knife_006", "apple_001")
    Image.new("RGB", (64, 48), (1, 2, 3)).save(pair_dir / "angle_00.png")
    Image.new("RGB", (64, 48), (4, 5, 6)).save(pair_dir / "angle_01.png")

    tiles_dir = tmp_path / "tiles"
    (tiles_dir / "knife").mkdir(parents=True)
    (tiles_dir / "apple").mkdir(parents=True)
    Image.new("RGB", (64, 48), (7, 8, 9)).save(tiles_dir / "knife" / "tile_006.png")
    Image.new("RGB", (64, 48), (10, 11, 12)).save(tiles_dir / "apple" / "tile_001.png")

    job = {
        "actor_inst": "knife_006", "recipient_inst": "apple_001",
        "actor_cat": "knife", "recipient_cat": "apple",
        "predicates": ["canCut"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
        "pair_dir": pair_dir,
    }
    client = _FakeClient({"canCut": {"type": "box cutter", "score": 2}})
    labeler = OmniLabeler(client=client, tiles_dir=tiles_dir)
    labeler._label_pair(client, job, job["predicates"])

    # a composed grid should now exist and be what was actually sent
    composed = pair_dir / "cancut_grid.png"
    assert composed.exists()
    sent_b64 = client.calls[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert sent_b64.endswith(_b64(composed))


def _b64(path):
    import base64
    return base64.b64encode(path.read_bytes()).decode()


def test_label_pair_uses_composed_grid_when_cancut_bundled_with_canpierce(tmp_path):
    # 2026-08-04: is_candidate_pair makes canPierce a candidate for every
    # knife x food pair that canCut is, so build_pairs ALWAYS bundles them
    # into one job/one call for real knife pairs -- the predicates == ["canCut"]
    # check here never actually fired in production, and the solo-tile
    # image (built specifically for clearer type/tip classification) was
    # never sent for any real knife pair. canPierce doesn't need a size
    # reference (tip sharpness only, not size fit), so it's safe to include
    # in the same relaxed trigger as canCut.
    from PIL import Image

    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "knife_006", "apple_001")
    Image.new("RGB", (64, 48), (1, 2, 3)).save(pair_dir / "angle_00.png")
    Image.new("RGB", (64, 48), (4, 5, 6)).save(pair_dir / "angle_01.png")

    tiles_dir = tmp_path / "tiles"
    (tiles_dir / "knife").mkdir(parents=True)
    (tiles_dir / "apple").mkdir(parents=True)
    Image.new("RGB", (64, 48), (7, 8, 9)).save(tiles_dir / "knife" / "tile_006.png")
    Image.new("RGB", (64, 48), (10, 11, 12)).save(tiles_dir / "apple" / "tile_001.png")

    job = {
        "actor_inst": "knife_006", "recipient_inst": "apple_001",
        "actor_cat": "knife", "recipient_cat": "apple",
        "predicates": ["canCut", "canPierce"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
        "pair_dir": pair_dir,
    }
    client = _FakeClient({
        "canCut": {"type": "kitchen knife", "score": 8},
        "canPierce": 8,
    })
    labeler = OmniLabeler(client=client, tiles_dir=tiles_dir)
    labeler._label_pair(client, job, job["predicates"])

    composed = pair_dir / "cancut_grid.png"
    assert composed.exists()
    sent_b64 = client.calls[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert sent_b64.endswith(_b64(composed))


def test_label_pair_falls_back_to_standard_grid_when_tiles_missing(tmp_path):
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "knife_006", "apple_001")
    # no angle_00/01.png, no tiles_dir set -- must not crash, must use grid_path
    job = {
        "actor_inst": "knife_006", "recipient_inst": "apple_001",
        "actor_cat": "knife", "recipient_cat": "apple",
        "predicates": ["canCut"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
        "pair_dir": pair_dir,
    }
    client = _FakeClient({"canCut": {"type": "kitchen knife", "score": 8}})
    labeler = OmniLabeler(client=client)  # tiles_dir defaults to None
    result = labeler._label_pair(client, job, job["predicates"])

    assert result["canCut"] == 8
    sent_b64 = client.calls[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert sent_b64.endswith(_b64(pair_dir / "grid.png"))


def test_label_pair_gives_cancut_enough_tokens_for_low_reasoning(tmp_path):
    # 2026-07-14 live bug: reasoning_effort="low" burns hidden reasoning
    # tokens the "minimal"-tuned 50-tokens-per-predicate budget didn't
    # account for -- every canCut call came back empty (finish_reason=
    # "length") until the budget was raised for this predicate specifically.
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "knife_006", "apple_001")
    job = {
        "actor_inst": "knife_006", "recipient_inst": "apple_001",
        "actor_cat": "knife", "recipient_cat": "apple",
        "predicates": ["canCut"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
        "pair_dir": pair_dir,
    }
    client = _FakeClient({"canCut": {"type": "box cutter", "score": 2}})
    labeler = OmniLabeler(client=client)
    labeler._label_pair(client, job, job["predicates"])

    assert client.calls[0]["max_completion_tokens"] >= 1000


def test_label_pair_uses_low_reasoning_for_cancut(tmp_path):
    # reasoning_effort="minimal" was found to misclassify box-cutter-type
    # knives as folding/pocket knives 100% of the time (2026-07-14 live
    # test); low/medium/high all classified correctly. canCut alone gets
    # bumped to "low" -- cheapest level that was reliably correct -- while
    # every other predicate keeps "minimal" (verified fine at that level).
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "knife_006", "apple_001")
    job = {
        "actor_inst": "knife_006", "recipient_inst": "apple_001",
        "actor_cat": "knife", "recipient_cat": "apple",
        "predicates": ["canCut"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
        "pair_dir": pair_dir,
    }
    client = _FakeClient({"canCut": {"type": "box cutter", "score": 2}})
    labeler = OmniLabeler(client=client)
    labeler._label_pair(client, job, job["predicates"])

    assert client.calls[0]["reasoning_effort"] == "low"


def test_label_pair_uses_low_reasoning_for_every_predicate(tmp_path):
    # 2026-08-03: "minimal" was found unreliable for canCut, canContain,
    # canSupport, and canPierce -- four out of five VLM-judged predicates,
    # one at a time, each only after a real labeling defect surfaced. Rather
    # than wait to discover the same failure mode for canPeel too, "low" is
    # now used for every predicate; "minimal" is retired.
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "dish_011", "durian_030")
    job = {
        "actor_inst": "dish_011", "recipient_inst": "durian_030",
        "actor_cat": "dish", "recipient_cat": "durian",
        "predicates": ["canPeel"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }
    client = _FakeClient({"canPeel": 2})
    labeler = OmniLabeler(client=client)
    labeler._label_pair(client, job, job["predicates"])

    assert client.calls[0]["reasoning_effort"] == "low"
    assert client.calls[0]["max_completion_tokens"] >= 1000


def test_label_pair_uses_low_reasoning_for_cancontain_and_cansupport(tmp_path):
    # 2026-08-01: E3 (36-pair human gold set) found "minimal" reasoning
    # produces confident wrong scores on canContain (bowl_011 x
    # broccolini_001: 0.89 for a stalk sticking out of a shallow bowl) and
    # canSupport (cup_037 x sweet_potato_001: 0.89 for a lidded mug). Same
    # fix pattern as canCut's 2026-07-14 bump: "minimal" isn't enough
    # reasoning for these two, "low" is.
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "dish_011", "durian_030")
    job = {
        "actor_inst": "dish_011", "recipient_inst": "durian_030",
        "actor_cat": "dish", "recipient_cat": "durian",
        "predicates": ["canContain", "canSupport"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }
    client = _FakeClient({
        "canContain": {"score": 7, "enclosure": "walls on all sides"},
        "canSupport": {"score": 2, "size_note": "food larger than surface"},
    })
    labeler = OmniLabeler(client=client)
    labeler._label_pair(client, job, job["predicates"])

    assert client.calls[0]["reasoning_effort"] == "low"
    assert client.calls[0]["max_completion_tokens"] >= 1000


def test_label_pair_uses_low_reasoning_for_canpierce(tmp_path):
    # 2026-08-03: canPierce moved from a hardcoded rule to a real VLM
    # judgment. Its whole question hinges on a subtle visual cue (a tapered
    # point vs. a blunt/rounded tip) -- the same class of fine-grained
    # judgment that pushed canCut/canContain/canSupport to "low" reasoning,
    # so canPierce gets the same treatment from the start rather than
    # discovering the same "minimal" failure mode again later.
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "fork_002", "apple_016")
    job = {
        "actor_inst": "fork_002", "recipient_inst": "apple_016",
        "actor_cat": "fork", "recipient_cat": "apple",
        "predicates": ["canPierce"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }
    client = _FakeClient({"canPierce": 9})
    labeler = OmniLabeler(client=client)
    labeler._label_pair(client, job, job["predicates"])

    assert client.calls[0]["reasoning_effort"] == "low"
    assert client.calls[0]["max_completion_tokens"] >= 1000


def test_label_pair_normalizes_cancut_dict_response_to_plain_score(tmp_path):
    # canCut's prompt asks for {"type": "...", "score": N} to make the model
    # articulate its knife-type classification (validated 2026-07-14: without
    # it, box-cutter-type meshes default to a lazy "5" instead of their
    # correct 2-3 cap) -- _label_pair must unwrap the dict, discarding type.
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "knife_006", "apple_001")

    job = {
        "actor_inst": "knife_006", "recipient_inst": "apple_001",
        "actor_cat": "knife", "recipient_cat": "apple",
        "predicates": ["canCut"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }
    client = _FakeClient({"canCut": {"type": "box cutter", "score": 2}})
    labeler = OmniLabeler(client=client)
    result = labeler._label_pair(client, job, job["predicates"])

    assert result["canCut"] == 2  # unwrapped, not the dict


def test_label_pair_makes_one_call_for_all_predicates(tmp_path):
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "dish_011", "durian_030")

    job = {
        "actor_inst": "dish_011", "recipient_inst": "durian_030",
        "actor_cat": "dish", "recipient_cat": "durian",
        "predicates": ["canContain", "canSupport"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }
    client = _FakeClient({"canContain": 7, "canSupport": 2})
    labeler = OmniLabeler(client=client)
    result = labeler._label_pair(client, job, job["predicates"])

    assert len(client.calls) == 1  # one call covers BOTH predicates
    assert result["canContain"] == 7
    assert result["canSupport"] == 2


def test_run_writes_one_record_per_predicate(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_011", "recipient": "durian_030"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_011", "durian_030")
    out_path = tmp_path / "labels.jsonl"

    client = _FakeClient({"canContain": 7, "canSupport": 2})
    labeler = OmniLabeler(client=client)
    labeler.run(pairs_json, renders_dir, out_path)

    lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    by_pred = {r["predicate"]: r for r in lines}
    assert by_pred["canContain"]["score"] == 7
    assert by_pred["canContain"]["rationale"] is None
    assert by_pred["canSupport"]["score"] == 2


def test_run_writes_auto_negative_for_flipped_container_without_api_call(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_011", "recipient": "durian_030"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_011", "durian_030", container_flipped=True)
    out_path = tmp_path / "labels.jsonl"

    client = _FakeClient({})  # would fail/mismatch if actually called
    labeler = OmniLabeler(client=client)
    labeler.run(pairs_json, renders_dir, out_path)

    assert len(client.calls) == 0  # no API call spent on a broken image
    lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2  # canContain + canSupport, no gap
    by_pred = {r["predicate"]: r for r in lines}
    assert by_pred["canContain"]["score"] == 0
    assert by_pred["canSupport"]["score"] == 0


def test_run_is_resumable(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_011", "recipient": "durian_030"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_011", "durian_030")
    out_path = tmp_path / "labels.jsonl"
    out_path.write_text(json.dumps({
        "actor": "dish_011", "recipient": "durian_030",
        "predicate": "canContain", "score": 7, "rationale": "fits fine",
    }) + "\n")

    client = _FakeClient({"canSupport": 2})
    labeler = OmniLabeler(client=client)
    labeler.run(pairs_json, renders_dir, out_path)

    assert len(client.calls) == 1  # only the missing predicate was requested
    lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    predicates = {r["predicate"] for r in lines}
    assert predicates == {"canContain", "canSupport"}


def test_label_pair_accumulates_token_usage(tmp_path):
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "dish_011", "durian_030")
    job = {
        "actor_inst": "dish_011", "recipient_inst": "durian_030",
        "actor_cat": "dish", "recipient_cat": "durian",
        "predicates": ["canContain"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }
    client = _FakeClient({"canContain": 7})  # _FakeClient reports 10 prompt / 5 completion tokens
    labeler = OmniLabeler(client=client)
    labeler._label_pair(client, job, job["predicates"])

    assert labeler.total_prompt_tokens == 10
    assert labeler.total_completion_tokens == 5
    assert labeler.total_reasoning_tokens == 0


class _FailingClient:
    """Raises a fixed exception on every call -- for retry/backoff tests."""
    def __init__(self, exc_factory, succeed_after=None, final_response=None):
        self.exc_factory = exc_factory
        self.succeed_after = succeed_after
        self.final_response = final_response
        self.calls = []

        class _Chat:
            def __init__(inner_self, outer):
                inner_self.completions = inner_self

            def create(inner_self, **kwargs):
                self.calls.append(kwargs)
                if self.succeed_after is not None and len(self.calls) > self.succeed_after:
                    class _Msg:
                        content = json.dumps(self.final_response)
                    class _Choice:
                        message = _Msg()
                        finish_reason = "stop"
                    class _TokDetails:
                        reasoning_tokens = 0
                    class _Usage:
                        prompt_tokens = 1
                        completion_tokens = 1
                        completion_tokens_details = _TokDetails()
                    class _Resp:
                        choices = [_Choice()]
                        usage = _Usage()
                    return _Resp()
                raise self.exc_factory()

        self.chat = _Chat(self)


def test_label_pair_fails_fast_on_permanent_4xx_without_exhausting_retries(tmp_path, monkeypatch):
    from openai import APIStatusError

    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "dish_011", "durian_030")
    job = {
        "actor_inst": "dish_011", "recipient_inst": "durian_030",
        "actor_cat": "dish", "recipient_cat": "durian",
        "predicates": ["canContain"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }

    def make_400():
        return APIStatusError(
            "bad request", response=_FakeHttpResponse(400), body=None,
        )

    monkeypatch.setattr("time.sleep", lambda s: None)
    client = _FailingClient(make_400)
    labeler = OmniLabeler(client=client, max_retries=5)
    result = labeler._label_pair(client, job, job["predicates"])

    assert result is None
    assert len(client.calls) == 1  # no retry burned on a non-429 4xx


def test_label_pair_retries_transient_error_then_succeeds(tmp_path, monkeypatch):
    renders_dir = tmp_path / "renders"
    pair_dir = _write_manifest(renders_dir, "dish_011", "durian_030")
    job = {
        "actor_inst": "dish_011", "recipient_inst": "durian_030",
        "actor_cat": "dish", "recipient_cat": "durian",
        "predicates": ["canContain"],
        "grid_path": pair_dir / "grid.png",
        "grid_rows": 2, "grid_cols": 2,
    }

    monkeypatch.setattr("time.sleep", lambda s: None)
    client = _FailingClient(
        lambda: ConnectionError("transient network blip"),
        succeed_after=2, final_response={"canContain": 6},
    )
    labeler = OmniLabeler(client=client, max_retries=5)
    result = labeler._label_pair(client, job, job["predicates"])

    assert result == {"canContain": 6}
    assert len(client.calls) == 3  # 2 failures + 1 success


class _FakeHttpResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}
        self.request = None


class _SlowFakeClient:
    """Like _FakeClient but each call takes real wall-clock time, so
    concurrency actually shortens total run time in a measurable way."""
    def __init__(self, response_json, delay=0.05):
        self.response_json = response_json
        self.delay = delay
        self.calls = []
        self._lock = __import__("threading").Lock()

        class _Chat:
            def __init__(inner_self, outer):
                inner_self.completions = inner_self

            def create(inner_self, **kwargs):
                import time as _t
                with self._lock:
                    self.calls.append(kwargs)
                _t.sleep(self.delay)

                class _Msg:
                    content = json.dumps(self.response_json)

                class _Choice:
                    message = _Msg()
                    finish_reason = "stop"

                class _TokDetails:
                    reasoning_tokens = 0

                class _Usage:
                    prompt_tokens = 10
                    completion_tokens = 5
                    completion_tokens_details = _TokDetails()

                class _Resp:
                    choices = [_Choice()]
                    usage = _Usage()

                return _Resp()

        self.chat = _Chat(self)


def test_run_with_multiple_workers_labels_all_pairs(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([
        {"actor": "dish_011", "recipient": "durian_030"},
        {"actor": "dish_012", "recipient": "durian_031"},
        {"actor": "dish_013", "recipient": "durian_032"},
        {"actor": "dish_014", "recipient": "durian_033"},
    ]))
    renders_dir = tmp_path / "renders"
    for i in range(11, 15):
        _write_manifest(renders_dir, f"dish_0{i}", f"durian_0{i+19}")
    out_path = tmp_path / "labels.jsonl"

    client = _SlowFakeClient({"canContain": 7, "canSupport": 2}, delay=0.05)
    labeler = OmniLabeler(client=client)
    labeler.run(pairs_json, renders_dir, out_path, max_workers=4)

    lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 8  # 4 pairs x 2 predicates each
    pairs_seen = {(r["actor"], r["recipient"]) for r in lines}
    assert len(pairs_seen) == 4


def test_run_with_multiple_workers_is_faster_than_sequential(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([
        {"actor": f"dish_0{i}", "recipient": f"durian_0{i+19}"} for i in range(11, 19)
    ]))
    renders_dir = tmp_path / "renders"
    for i in range(11, 19):
        _write_manifest(renders_dir, f"dish_0{i}", f"durian_0{i+19}")
    out_path = tmp_path / "labels.jsonl"

    client = _SlowFakeClient({"canContain": 7, "canSupport": 2}, delay=0.05)
    labeler = OmniLabeler(client=client)

    start = time.monotonic()
    labeler.run(pairs_json, renders_dir, out_path, max_workers=8)
    elapsed = time.monotonic() - start

    # sequential would take ~8 * 0.05 = 0.4s; concurrent should be well under
    assert elapsed < 0.3


def test_run_concurrent_reports_usage_to_rate_limiter(tmp_path):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([
        {"actor": "dish_011", "recipient": "durian_030"},
        {"actor": "dish_012", "recipient": "durian_031"},
    ]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_011", "durian_030")
    _write_manifest(renders_dir, "dish_012", "durian_031")
    out_path = tmp_path / "labels.jsonl"

    client = _FakeClient({"canContain": 7, "canSupport": 2})
    limiter = TokenRateLimiter(tokens_per_minute=1_000_000)
    labeler = OmniLabeler(client=client, rate_limiter=limiter)
    labeler.run(pairs_json, renders_dir, out_path, max_workers=2)

    # 2 pairs x 10 prompt tokens (per _FakeClient) reported to the limiter
    assert limiter.current_tokens_per_minute() > 0


def test_dry_run_does_not_call_client(tmp_path, capsys):
    pairs_json = tmp_path / "pairs.json"
    pairs_json.write_text(json.dumps([{"actor": "dish_011", "recipient": "durian_030"}]))
    renders_dir = tmp_path / "renders"
    _write_manifest(renders_dir, "dish_011", "durian_030")
    out_path = tmp_path / "labels.jsonl"

    client = _FakeClient({})
    labeler = OmniLabeler(client=client)
    labeler.run(pairs_json, renders_dir, out_path, dry_run=True)

    assert client.calls == []
    assert not out_path.exists()
