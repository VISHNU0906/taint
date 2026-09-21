import pytest
from taint.replay import evaluate_rows

def outcome(**changes):
    row = dict(id="one", family="a", split="test", source="fixture", model="fixture",
               score=0.9, attack_present=True, attack_succeeded=True, task_succeeded=False)
    return row | changes

def test_separate_utility_and_attack_success():
    report = evaluate_rows([outcome(), outcome(id="two", family="b", score=0.1,
        attack_present=False, attack_succeeded=False, task_succeeded=True)])
    assert report["clean_task_success"] == report["observed_attack_success"] == 1
    assert report["clean_flag_rate"] == 0

def test_leakage_across_splits_rejected():
    with pytest.raises(ValueError, match="across splits"):
        evaluate_rows([outcome(), outcome(id="two", split="train")])

@pytest.mark.parametrize("change", [{"score": float("nan")}, {"score": True},
    {"attack_present": "false"}, {"split": "unknown"}, {"source": ""},
    {"attack_present": False}])
def test_invalid_outcome(change):
    with pytest.raises(ValueError):
        evaluate_rows([outcome(**change)])

def test_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_rows([outcome(), outcome()])

def test_missing_class_is_not_perfect_score():
    report = evaluate_rows([outcome()])
    assert report["clean_task_success"] is None
