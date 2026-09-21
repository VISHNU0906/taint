import pytest
from taint.study import study, calibration


def rows():
    return [{"id": f"{split}-{i}", "family": f"{split}-family-{i}", "split": split,
             "source": "fixture", "model": "fixture", "score": score,
             "attack_present": attack, "attack_succeeded": False, "task_succeeded": True}
            for split in ["validation", "test"]
            for i, (score, attack) in enumerate([(0.1, False), (0.3, False), (0.7, True), (0.9, True)])]


def test_test_scores_cannot_tune_threshold():
    data = rows()
    before = study(data, repeats=20)
    for row in data:
        if row["split"] == "test":
            row["score"] = 1 - row["score"]
    after = study(data, repeats=20)
    assert before["selection"] == after["selection"]
    assert before["held_out"]["recall"] != after["held_out"]["recall"]


def test_no_family_overlap():
    data = rows()
    data[-1]["family"] = data[0]["family"]
    with pytest.raises(ValueError, match="across splits"):
        study(data)


def test_perfect_probabilities():
    assert calibration([{"score": 0, "attack_present": False}, {"score": 1, "attack_present": True}])["brier"] == 0


def test_bootstrap_reproducible():
    assert study(rows(), repeats=50, seed=3) == study(rows(), repeats=50, seed=3)


def test_infeasible_budget():
    data = rows()
    data[0]["score"] = 1
    with pytest.raises(ValueError, match="no threshold"):
        study(data, max_clean_flag_rate=0)
