from affdet.eval.baseline_vlm_prompts import (
    CATEGORY_VOCAB, build_baseline_system_prompt, build_baseline_user_prompt,
    parse_baseline_response,
)


def test_category_vocab_matches_curation_taxonomy():
    # Spot-check a few known categories from scripts/omni/curation.json rather
    # than pinning the whole list, so this doesn't break on taxonomy tweaks.
    assert "knife" in CATEGORY_VOCAB
    assert "bowl" in CATEGORY_VOCAB
    assert "peeler" in CATEGORY_VOCAB
    assert len(CATEGORY_VOCAB) >= 50


def test_system_prompt_includes_all_six_predicates_calibration():
    prompt = build_baseline_system_prompt()
    for term in ("canCut", "canPierce", "canPeel", "canContain",
                 "canSupport", "canPourInto"):
        assert term in prompt


def test_system_prompt_includes_category_vocab():
    prompt = build_baseline_system_prompt()
    assert "knife" in prompt
    assert "bowl" in prompt


def test_system_prompt_gives_real_spout_definition_for_canpourinto():
    prompt = build_baseline_system_prompt()
    assert "spout" in prompt.lower()


def test_user_prompt_asks_for_objects_and_all_pairs():
    prompt = build_baseline_user_prompt()
    assert "object" in prompt.lower()
    assert "ordered pair" in prompt.lower() or "every pair" in prompt.lower()


def test_parse_baseline_response_extracts_objects_and_pairs():
    raw = """{
        "objects": [{"id": "knife_1", "category": "knife"},
                    {"id": "apple_1", "category": "apple"}],
        "affordances": [
            {"actor": "knife_1", "recipient": "apple_1", "predicate": "canCut", "verdict": 1},
            {"actor": "knife_1", "recipient": "apple_1", "predicate": "canPierce", "verdict": 1}
        ]
    }"""
    parsed = parse_baseline_response(raw)
    assert parsed["objects"] == [{"id": "knife_1", "category": "knife"},
                                  {"id": "apple_1", "category": "apple"}]
    assert len(parsed["affordances"]) == 2
    assert parsed["affordances"][0]["predicate"] == "canCut"


def test_parse_baseline_response_strips_markdown_code_fence():
    raw = "```json\n{\"objects\": [], \"affordances\": []}\n```"
    parsed = parse_baseline_response(raw)
    assert parsed == {"objects": [], "affordances": []}
