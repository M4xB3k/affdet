from affdet.experiment.ledger import append_run, load_runs, git_sha


def test_append_and_load_roundtrip(tmp_path):
    p = tmp_path/"runs.jsonl"
    append_run(p, {"run_id": "a", "metrics": {"macro_f1": 0.88}})
    append_run(p, {"run_id": "b", "metrics": {"macro_f1": 0.90}})
    rows = load_runs(p)
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert rows[1]["metrics"]["macro_f1"] == 0.90


def test_load_missing_file_returns_empty(tmp_path):
    assert load_runs(tmp_path/"none.jsonl") == []


def test_git_sha_is_str():
    s = git_sha()
    assert isinstance(s, str) and len(s) > 0
