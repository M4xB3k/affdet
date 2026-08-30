"""DEGENERATE_PREDICATES is the macro-average exclusion list.

A predicate belongs here only while it is structurally unscoreable -- i.e. the
corpus gives it a single verdict everywhere, so a model can score F1=1.0 by
answering the same way every time without looking at the image. Averaging such
a predicate into the headline macro inflates it for free.

canPierce was on this list because the v3 corpus had ZERO negatives for it
(every target came from a hardcoded `return 1.0` in pair_vlm_resolver, fixed in
b80441f). The v4 relabel gives it 1,092 real negatives, so the justification no
longer holds and it must be scored like any other predicate.

This is a REPORTING switch only -- it is consumed by compute_metrics /
write_metrics_csv after predict(), and never by training. See
test_exclusion_is_reporting_only_not_training below.
"""
from pathlib import Path

import affdet.experiment.runner as runner


def test_canpierce_is_no_longer_excluded_from_the_macro():
    # v4 gives canPierce real negatives; excluding it now DROPS a working
    # predicate from the headline instead of protecting it.
    assert "canPierce" not in runner.DEGENERATE_PREDICATES


def test_no_predicate_is_excluded_by_default():
    # Nothing else in the corpus is single-verdict. If a future predicate
    # becomes degenerate, adding it here should be a deliberate, documented
    # act -- not something that quietly accumulates.
    assert runner.DEGENERATE_PREDICATES == []


def test_exclusion_is_reporting_only_not_training():
    """Guards the property that makes this list cheap to change: flipping it
    must never require retraining. If DEGENERATE_PREDICATES ever leaks into
    the training path, a reporting tweak silently invalidates checkpoints."""
    train_src = Path(runner.__file__).parent.parent / "model" / "train.py"
    dataset_src = Path(runner.__file__).parent.parent / "model" / "dataset.py"
    for src in (train_src, dataset_src):
        text = src.read_text()
        assert "DEGENERATE_PREDICATES" not in text, src
        assert "macro_exclude" not in text, src
