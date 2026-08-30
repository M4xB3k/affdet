"""The E7 inference scripts must not carry a default checkpoint.

History, because the fix was made twice:

1. `scripts/e7_student_infer.py` defaulted to a **v3** run. `labels_all_merged_v3`
   carried canPierce as a hardcoded constant, so that checkpoint emits a flat 0.5
   for every canPierce pair. Running the script without `--ckpt` during E8
   silently produced exactly that, and the constant output was mistaken for a
   model defect (`E8_DIRECT_VLM_TRAINING.md`).

2. The default was re-pointed at a v4 run, which reproduced the same failure one
   corpus later: E9 trained v5 students, and an arm-E run that omitted `--ckpt`
   would have emitted v4 predictions labelled as v5.

Pinning the default to "the current corpus" fails every time the corpus advances.
These tests therefore assert the checkpoint is **required**, not merely modern.
"""
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
CKPT_SCRIPTS = [
    SCRIPTS / "e7_student_infer.py",
    SCRIPTS / "e7_student_arm_e.py",
    SCRIPTS / "e7_calibrate.py",
]


def _ckpt_argument(path: Path) -> str:
    """The `--ckpt` add_argument call, through to its closing paren."""
    text = path.read_text()
    i = text.index('"--ckpt"')
    depth, j = 0, text.rindex("(", 0, i)
    for j in range(j, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                break
    return text[i:j]


def test_ckpt_is_required_everywhere():
    """A missing --ckpt must be an error, not a silent corpus-version choice."""
    for path in CKPT_SCRIPTS:
        arg = _ckpt_argument(path)
        assert "required=True" in arg, f"{path.name}: --ckpt must be required"
        assert "default=" not in arg, (
            f"{path.name}: --ckpt must not have a default -- pinning it to a "
            "corpus version has silently produced wrong predictions twice")


def test_no_module_level_ckpt_constant():
    """A CKPT constant is how the default came back the second time."""
    for path in CKPT_SCRIPTS:
        assert not re.search(r"^CKPT\s*=", path.read_text(), re.M), (
            f"{path.name}: module-level CKPT constant reintroduces the default")


def test_scripts_warn_about_the_corpus_version_artefact():
    """The next reader must be told why the checkpoint matters."""
    text = (SCRIPTS / "e7_student_infer.py").read_text()
    assert "v3" in text and "canPierce" in text


def test_calibrate_seed_is_not_hardcoded():
    """The val split is drawn from `seed`, so it must track the checkpoint.

    Seed 1's val split shares only 268 of 1912 scenes with seed 0's. Calibrating
    a seed-1 checkpoint against seed 0's split fits thresholds on scenes that
    checkpoint trained on -- leakage, and it happened: the committed
    e9_thresholds_v5_s{0,1,2}.json all record seed 0.
    """
    text = (SCRIPTS / "e7_calibrate.py").read_text()
    assert re.search(r"seed\s*=\s*args\.seed", text), (
        "Inferencer(seed=...) must come from --seed, not a literal")
    assert re.search(r'"seed":\s*args\.seed', text), (
        "the fit_on metadata must record the real seed, not a literal")
