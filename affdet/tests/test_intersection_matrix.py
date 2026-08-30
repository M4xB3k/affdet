from affdet.intersection_matrix import (
    is_candidate_pair,
    candidate_pairs_for_scene,
    OBJECT_CLASSES,
    TIER1_PREDICATES,
)


def test_knife_canCut_food():
    assert is_candidate_pair("knife", "apple", "canCut") is True
    assert is_candidate_pair("knife", "banana", "canCut") is True


def test_food_cannot_be_actor_for_canCut():
    assert is_candidate_pair("banana", "knife", "canCut") is False


def test_scissor_has_no_affordances():
    for pred in TIER1_PREDICATES:
        assert is_candidate_pair("scissor", "apple", pred) is False


def test_knife_and_fork_canPierce_food():
    assert is_candidate_pair("knife", "tomato", "canPierce") is True
    assert is_candidate_pair("fork", "tomato", "canPierce") is True


def test_peeler_canPeel_any_food():
    # canPeel recipients are no longer hard-filtered to a fixed hard-skinned
    # subset -- every food is a structural candidate, and the VLM judges
    # per-instance peelability (e.g. bread) rather than a hand-picked list
    # deciding it up front.
    assert is_candidate_pair("peeler", "apple", "canPeel") is True
    assert is_candidate_pair("peeler", "corn", "canPeel") is True
    assert is_candidate_pair("peeler", "bread", "canPeel") is True


def test_corn_is_a_candidate_for_both_cut_and_pierce():
    # Both exclusions are gone. canCut's was removed first (a knife genuinely
    # can cut corn); canPierce's followed once the predicate stopped being a
    # hardcoded rule (72d0ef6) and became a VLM judgment -- whether a tip can
    # enter a cob is a visual call, and pre-excluding corn removed the only
    # structural negative canPierce had.
    assert is_candidate_pair("knife", "corn", "canCut") is True
    assert is_candidate_pair("knife", "corn", "canPierce") is True
    assert is_candidate_pair("fork", "corn", "canPierce") is True


def test_canPierce_is_a_candidate_for_other_food():
    assert is_candidate_pair("knife", "tomato", "canPierce") is True
    assert is_candidate_pair("fork", "apple", "canPierce") is True


def test_open_containers_canContain_food():
    assert is_candidate_pair("bowl", "strawberry", "canContain") is True
    assert is_candidate_pair("cup", "apple", "canContain") is True
    assert is_candidate_pair("dish", "tomato", "canContain") is True


def test_pourables_canPourInto_open_containers():
    assert is_candidate_pair("teapot", "bowl", "canPourInto") is True
    assert is_candidate_pair("kettle", "cup", "canPourInto") is True
    assert is_candidate_pair("kettle", "apple", "canPourInto") is False


def test_bottle_is_not_a_pour_actor():
    # bottle pours from nothing (all sealed) -> never an actor
    assert is_candidate_pair("bottle", "dish", "canPourInto") is False
    assert is_candidate_pair("bottle", "bowl", "canPourInto") is False


def test_teapot_and_vase_can_be_pour_recipients():
    # relational: an open-top vessel can be poured INTO (per-instance gated later)
    assert is_candidate_pair("kettle", "teapot", "canPourInto") is True
    assert is_candidate_pair("teapot", "vase", "canPourInto") is True


def test_receive_only_classes_are_not_pour_actors():
    for cls in ("bowl", "dish", "pan", "red_wine_glass", "cup"):
        assert is_candidate_pair(cls, "bowl", "canPourInto") is False


def test_kettle_thermos_not_valid_recipients():
    # kettle/thermos pour out but cannot be poured into
    assert is_candidate_pair("teapot", "kettle", "canPourInto") is False
    assert is_candidate_pair("teapot", "thermos", "canPourInto") is False


def test_knife_cannot_contain():
    assert is_candidate_pair("knife", "apple", "canContain") is False


def test_open_containers_canSupport_food():
    assert is_candidate_pair("bowl", "apple", "canSupport") is True
    assert is_candidate_pair("pan", "tomato", "canSupport") is True


def test_candidate_pairs_for_scene():
    classes = ["knife", "fork", "bowl", "apple", "strawberry", "kettle"]
    pairs = candidate_pairs_for_scene(classes)
    assert ("knife", "apple", "canCut") in pairs
    assert ("fork", "apple", "canPierce") in pairs
    assert ("bowl", "strawberry", "canContain") in pairs
    assert ("kettle", "bowl", "canPourInto") in pairs
    assert ("apple", "knife", "canCut") not in pairs


def test_6_predicates():
    assert set(TIER1_PREDICATES) == {
        "canCut", "canPierce", "canPeel", "canContain", "canSupport", "canPourInto",
    }


def test_object_classes_contains_key_categories():
    expected = {
        "knife", "peeler", "fork", "scissor",
        "bowl", "cup", "dish", "pan", "red_wine_glass",
        "teapot", "vase", "kettle", "bottle", "thermos",
        "apple", "banana", "tomato", "watermelon",
    }
    assert expected.issubset(set(OBJECT_CLASSES))
