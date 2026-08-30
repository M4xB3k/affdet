import json
from pathlib import Path

from affdet.label.resolved_instances import (
    build_resolved, PAN_DROP, load_effective_class_map, effective_class,
    load_dropped_set,
)


def _curation():
    return {
        "_meta": {"format": 3},
        "_suitability": {"knife": {"canCut": ["apple"]}},
        "knife": {"drop": ["007"], "recategorize": {"011": "peeler"}},
        "bottle": {"drop": ["030"], "open_closed": {"closed": ["001"]}},
        "teapot": {"open_closed": {"open": ["014"], "closed": ["001"]}},
        "pan": {"open_closed": {"closed": ["002"]}},
    }


def test_ignores_underscore_keys():
    r = build_resolved(_curation())
    assert not any(k.startswith("_") for k in r)


def test_recategorized_knife_becomes_peeler():
    r = build_resolved(_curation())
    assert r["knife_011"]["effective_class"] == "peeler"


def test_curation_drop_marked():
    r = build_resolved(_curation())
    assert r["knife_007"]["dropped"] is True
    assert r["knife_007"]["reason"] == "curation_drop"


def test_pan_artifact_dropped():
    r = build_resolved(_curation())
    assert r["pan_001"]["dropped"] is True
    assert r["pan_001"]["reason"] == "multiview_vlm_artifact"
    assert r["pan_003"]["dropped"] is True


def test_pan_drop_constant():
    assert PAN_DROP == {"pan_001", "pan_003"}


# --- pourable_from / pourable_into (relational canPourInto, class-rule table) ---
# The truth table lives in resolved_instances.py; these pin its per-class rules.

def _pour_curation():
    # Every pour-relevant class must appear so build_resolved emits its instances.
    # Only the numbers we assert on need to be present.
    return {
        "bottle": {"open_closed": {"closed": ["001", "045"]}},
        "thermos": {"open_closed": {"closed": ["001", "002", "003", "004", "005", "006"]}},
        "kettle": {"open_closed": {"closed": ["001"]}},
        "teapot": {"open_closed": {"open": ["014"], "closed": ["001", "023", "035", "037", "038"]}},
        "vase": {"open_closed": {"open": ["004", "012"], "closed": ["011", "001"]}},
        "cup": {"open_closed": {"open": ["001", "036"], "closed": ["002", "009", "033", "038", "039"]}},
        "bowl": {"open_closed": {"open": ["001"], "closed": ["009"]}},
        "dish": {"open_closed": {"open": ["001"]}},
        "pan": {"open_closed": {"open": ["004"]}},
        "red_wine_glass": {"open_closed": {"open": ["001"]}},
    }


def test_bottle_neither_from_nor_into():
    r = build_resolved(_pour_curation())
    assert r["bottle_001"]["pourable_from"] is False
    assert r["bottle_001"]["pourable_into"] is False


def test_kettle_from_only():
    r = build_resolved(_pour_curation())
    assert r["kettle_001"]["pourable_from"] is True
    assert r["kettle_001"]["pourable_into"] is False


def test_thermos_from_only_2_4_5():
    # Only instances 002/004/005 are verified pourable (user-corrected
    # 2026-07-12: was wrongly "all" -- most thermos lids don't actually open).
    r = build_resolved(_pour_curation())
    for num in ("002", "004", "005"):
        assert r[f"thermos_{num}"]["pourable_from"] is True, num
        assert r[f"thermos_{num}"]["pourable_into"] is False, num
    for num in ("001", "003", "006"):
        assert r[f"thermos_{num}"]["pourable_from"] is False, num


def test_teapot_all_from_but_into_only_listed():
    r = build_resolved(_pour_curation())
    # all teapots pour FROM (spout; lid irrelevant)
    assert r["teapot_023"]["pourable_from"] is True
    assert r["teapot_014"]["pourable_from"] is True
    # only 14,35,36,37,38 can be poured INTO
    assert r["teapot_014"]["pourable_into"] is True
    assert r["teapot_037"]["pourable_into"] is True
    assert r["teapot_023"]["pourable_into"] is False  # lidded -> not into
    assert r["teapot_001"]["pourable_into"] is False


def test_vase_from_only_4_12_into_all_but_11():
    r = build_resolved(_pour_curation())
    assert r["vase_004"]["pourable_from"] is True
    assert r["vase_012"]["pourable_from"] is True
    assert r["vase_001"]["pourable_from"] is False   # no spout
    assert r["vase_001"]["pourable_into"] is True
    assert r["vase_012"]["pourable_into"] is True
    assert r["vase_011"]["pourable_into"] is False   # has a lid


def test_cup_into_except_closed_never_from():
    r = build_resolved(_pour_curation())
    assert r["cup_001"]["pourable_from"] is False
    assert r["cup_001"]["pourable_into"] is True
    assert r["cup_036"]["pourable_into"] is True     # user-confirmed open
    for closed in ("002", "009", "033", "038", "039"):
        assert r[f"cup_{closed}"]["pourable_into"] is False


def test_receive_only_classes_whole_class_into():
    r = build_resolved(_pour_curation())
    for cls, num in [("bowl", "009"), ("dish", "001"), ("pan", "004"),
                     ("red_wine_glass", "001")]:
        # whole class receives regardless of open_closed (bowl_009 is 'closed'
        # in curation but still a valid recipient)
        assert r[f"{cls}_{num}"]["pourable_into"] is True, f"{cls}_{num}"
        assert r[f"{cls}_{num}"]["pourable_from"] is False


def test_list_shaped_recategorize_normalized():
    curation = {
        "bread": {"recategorize": [{"tile": "093", "to": "cake"}]},
    }
    r = build_resolved(curation)
    assert r["bread_093"]["effective_class"] == "cake"


def test_pour_class_flags_come_from_class_rule_not_open_closed():
    # A teapot instance not in any open_closed list still gets class-rule flags:
    # all teapots pour FROM; into only for the listed set (099 not listed -> False).
    curation = {
        "teapot": {
            "drop": ["099"],
            "open_closed": {"open": ["014"], "closed": ["001"]},
        },
    }
    r = build_resolved(curation)
    assert r["teapot_099"]["pourable_from"] is True
    assert r["teapot_099"]["pourable_into"] is False


def test_recategorized_to_pour_class_gets_pour_flags():
    curation = {
        "china": {"recategorize": {"024": "vase"}},
    }
    r = build_resolved(curation)
    assert r["china_024"]["effective_class"] == "vase"
    # vase: from only if 4/12 (024 not) -> False; into all-but-11 -> True
    assert r["china_024"]["pourable_from"] is False
    assert r["china_024"]["pourable_into"] is True


def test_load_effective_class_map_reads_written_file(tmp_path):
    resolved = build_resolved(_curation())
    out = tmp_path / "resolved_instances.json"
    out.write_text(json.dumps(resolved))
    m = load_effective_class_map(out)
    assert m["knife_011"] == "peeler"


def test_effective_class_with_map_uses_mapped_value(tmp_path):
    resolved = build_resolved(_curation())
    out = tmp_path / "resolved_instances.json"
    out.write_text(json.dumps(resolved))
    m = load_effective_class_map(out)
    assert effective_class("knife_011", "knife", m) == "peeler"


def test_effective_class_with_no_map_falls_back_to_obj_class():
    assert effective_class("knife_011", "knife", None) == "knife"


def test_load_dropped_set_includes_curation_and_artifact_drops(tmp_path):
    # knife_007 is a curation.json drop; pan_001/pan_003 are the separate
    # multiview-VLM-artifact drop (PAN_DROP) -- both reasons must end up in
    # the same dropped set, since this is meant to be the single place any
    # renderer checks "is this instance usable at all".
    resolved = build_resolved(_curation())
    out = tmp_path / "resolved_instances.json"
    out.write_text(json.dumps(resolved))
    dropped = load_dropped_set(out)
    assert "knife_007" in dropped
    assert "pan_001" in dropped
    assert "pan_003" in dropped


def test_load_dropped_set_excludes_kept_instances(tmp_path):
    resolved = build_resolved(_curation())
    out = tmp_path / "resolved_instances.json"
    out.write_text(json.dumps(resolved))
    dropped = load_dropped_set(out)
    assert "knife_011" not in dropped  # recategorized, not dropped


def test_effective_class_unmapped_id_falls_back_to_obj_class():
    m = {"knife_011": "peeler"}
    assert effective_class("apple_1", "apple", m) == "apple"


def test_runs_on_real_curation_json():
    curation_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "omni" / "curation.json"
    )
    curation = json.loads(curation_path.read_text())
    resolved = build_resolved(curation)
    assert resolved
    assert resolved["knife_011"]["effective_class"] == "peeler"
    assert resolved["pan_001"]["dropped"] is True
