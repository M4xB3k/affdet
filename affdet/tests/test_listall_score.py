"""E7 list-all arm — scoring volunteered output against the human labels.

The list-all arm returns a set of pairs the VLM CHOSE to report, so the join
back to the 593 human judgements is partial by construction. Three quantities
carry the finding:

  omission      human judgements the VLM never mentioned (the headline)
  hallucination pairs it reported that the structural gate calls impossible
  covered       accuracy/F1/MAE on the intersection only

Covered-subset metrics flatter the VLM: it is scored only on what it chose to
answer, and omitting the hard cases raises the score on what remains. So every
covered number is reported alongside two explicit imputations for the missing
rows -- omissions as negative, and omissions excluded. They bound the answer.
Picking one silently would beg the question the arm exists to ask.
"""
from affdet.eval.listall_score import (
    coverage_report,
    hallucination_report,
    score_covered,
    score_omissions_as_negative,
    aggregate_passes,
)

HUMAN_POS = 5


def _human():
    """Four judgements: three positive, one negative."""
    return {
        ("s1", "knife_1", "apple_1", "canCut"): 8,
        ("s1", "bowl_1", "apple_1", "canContain"): 9,
        ("s1", "peeler_1", "apple_1", "canPeel"): 7,
        ("s1", "cup_1", "melon_1", "canContain"): 1,
    }


# --------------------------------------------------------------------------
# Omission — the headline
# --------------------------------------------------------------------------

def test_omission_rate_counts_human_rows_the_vlm_never_mentioned():
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    rep = coverage_report(vlm, _human())
    assert rep["n_human"] == 4
    assert rep["n_covered"] == 1
    assert rep["n_omitted"] == 3
    assert rep["omission_rate"] == 0.75


def test_omission_rate_is_zero_when_everything_is_mentioned():
    vlm = {k: 5.0 for k in _human()}
    assert coverage_report(vlm, _human())["omission_rate"] == 0.0


def test_omission_is_about_being_mentioned_not_about_being_right():
    """A pair the VLM reported as 0 was still reported. Scoring it wrongly is
    an accuracy question; never raising it is an omission. Conflating them
    would hide the distinction the arm exists to measure."""
    vlm = {k: 0.0 for k in _human()}
    assert coverage_report(vlm, _human())["omission_rate"] == 0.0


def test_omission_rate_per_predicate():
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    per = coverage_report(vlm, _human())["per_predicate"]
    assert per["canCut"]["omission_rate"] == 0.0
    assert per["canPeel"]["omission_rate"] == 1.0
    assert per["canContain"]["n_human"] == 2
    assert per["canContain"]["n_omitted"] == 2


def test_omission_rate_split_by_human_polarity():
    """Omitting a pair the human called positive is the damaging case: recall
    becomes unmeasurable. Omitting a human-negative is closer to agreement --
    the VLM declined to claim an affordance that is not there."""
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    rep = coverage_report(vlm, _human())
    assert rep["n_omitted_positive"] == 2      # canContain(bowl), canPeel
    assert rep["n_omitted_negative"] == 1      # cup/melon
    assert rep["omission_rate_positive"] == 2 / 3


def test_coverage_report_handles_empty_vlm_output():
    rep = coverage_report({}, _human())
    assert rep["omission_rate"] == 1.0
    assert rep["n_covered"] == 0


# --------------------------------------------------------------------------
# Hallucination — against the structural gate
# --------------------------------------------------------------------------

def _cls():
    return {"knife_1": "knife", "apple_1": "apple", "bowl_1": "bowl",
            "peeler_1": "peeler", "cup_1": "cup", "melon_1": "watermelon"}


def test_hallucination_counts_structurally_impossible_reported_pairs():
    """apple cutting a knife is not a candidate under any labelling."""
    vlm = {
        ("s1", "knife_1", "apple_1", "canCut"): 8.0,     # fine
        ("s1", "apple_1", "knife_1", "canCut"): 6.0,     # impossible
    }
    rep = hallucination_report(vlm, _human(), _cls())
    assert rep["n_reported"] == 2
    assert rep["n_impossible"] == 1
    assert rep["hallucination_rate"] == 0.5


def test_plausible_but_unlabelled_is_reported_separately_not_charged():
    """A structurally valid pair the labeller simply never judged is not a
    hallucination. Charging it would overstate the rate and break comparability
    with the student's nonsense_positives, which uses the same convention."""
    vlm = {
        ("s1", "knife_1", "apple_1", "canCut"): 8.0,      # labelled
        ("s1", "knife_1", "apple_1", "canPierce"): 7.0,   # valid, unlabelled
    }
    rep = hallucination_report(vlm, _human(), _cls())
    assert rep["n_impossible"] == 0
    assert rep["hallucination_rate"] == 0.0
    assert rep["n_plausible_unlabelled"] == 1


def test_unknown_object_ids_are_counted_as_unresolvable():
    vlm = {("s1", "ghost_9", "apple_1", "canCut"): 8.0}
    rep = hallucination_report(vlm, _human(), _cls())
    assert rep["n_unresolvable"] == 1


# --------------------------------------------------------------------------
# Covered subset + the two imputations
# --------------------------------------------------------------------------

def test_score_covered_ignores_omitted_rows_entirely():
    """'Excluded' imputation: the denominator is the intersection."""
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    got = score_covered(vlm, _human())
    assert got["n"] == 1
    assert got["accuracy"] == 1.0
    assert got["mae"] == 0.0


def test_score_covered_mae_uses_the_raw_score_difference():
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 5.0}   # human 8
    assert score_covered(vlm, _human())["mae"] == 3.0


def test_omissions_as_negative_charges_missed_positives_as_false_negatives():
    """The pessimistic bound: not mentioning a pair == scoring it 0."""
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    got = score_omissions_as_negative(vlm, _human())
    assert got["n"] == 4
    # tp=1 (canCut); fn=2 (the two omitted positives); the omitted negative
    # becomes a true negative, since "not mentioned" agrees with "not an
    # affordance".
    assert got["recall"] == 1 / 3
    assert got["precision"] == 1.0
    assert got["accuracy"] == 0.5


def test_omissions_as_negative_scores_omitted_rows_as_zero_for_mae():
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    got = score_omissions_as_negative(vlm, _human())
    # |8-8|=0, |0-9|=9, |0-7|=7, |0-1|=1  -> 17/4
    assert got["mae"] == 17 / 4


def test_the_two_imputations_actually_differ():
    """If these ever coincide the report is not bounding anything, and the
    'report both' requirement has quietly become decorative."""
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    assert (score_covered(vlm, _human())["recall"]
            != score_omissions_as_negative(vlm, _human())["recall"])


def test_covered_metrics_are_computed_per_predicate_too():
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0,
           ("s1", "cup_1", "melon_1", "canContain"): 2.0}
    per = score_covered(vlm, _human(), per_predicate=True)["per_predicate"]
    assert per["canCut"]["n"] == 1
    assert per["canContain"]["n"] == 1
    assert per["canContain"]["accuracy"] == 1.0   # both call it negative


def test_covered_subset_can_flatter_the_vlm():
    """Documents the reason both imputations are mandatory: answering only the
    easy rows yields a perfect covered score next to 75 % omission."""
    vlm = {("s1", "knife_1", "apple_1", "canCut"): 8.0}
    assert score_covered(vlm, _human())["accuracy"] == 1.0
    assert coverage_report(vlm, _human())["omission_rate"] == 0.75


# --------------------------------------------------------------------------
# Multi-pass aggregation (median, matching the existing arms)
# --------------------------------------------------------------------------

def test_aggregate_passes_takes_the_median_score_per_triple():
    passes = [
        {("s1", "knife_1", "apple_1", "canCut"): 8.0},
        {("s1", "knife_1", "apple_1", "canCut"): 2.0},
        {("s1", "knife_1", "apple_1", "canCut"): 7.0},
    ]
    assert aggregate_passes(passes)[("s1", "knife_1", "apple_1", "canCut")] == 7.0


def test_aggregate_passes_keeps_a_pair_reported_in_a_minority_of_passes():
    """Aggregation must not become a second, hidden omission filter: a pair
    mentioned once out of three is still a pair the VLM volunteered, and the
    per-pass mention count is what the stability discussion needs."""
    passes = [
        {("s1", "a", "b", "canCut"): 8.0},
        {},
        {},
    ]
    agg = aggregate_passes(passes)
    assert ("s1", "a", "b", "canCut") in agg


def test_aggregate_passes_reports_mention_counts():
    passes = [
        {("s1", "a", "b", "canCut"): 8.0},
        {("s1", "a", "b", "canCut"): 6.0},
        {},
    ]
    _, counts = aggregate_passes(passes, with_counts=True)
    assert counts[("s1", "a", "b", "canCut")] == 2


def test_aggregate_passes_median_of_two_averages_them():
    passes = [{("s1", "a", "b", "canCut"): 4.0},
              {("s1", "a", "b", "canCut"): 6.0}]
    assert aggregate_passes(passes)[("s1", "a", "b", "canCut")] == 5.0


def test_aggregate_passes_on_no_passes_is_empty():
    assert aggregate_passes([]) == {}
