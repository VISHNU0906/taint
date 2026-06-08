"""Tests for the three detectors: each returns valid scores; canary is honest.

Asserts the required contract (every detector returns one [0,1] score per trace)
plus the integrity properties that make the benchmark credible: the canary does
not hallucinate injections into clean traces, and no detector reads the label.
"""

from __future__ import annotations

from taint.dataset import Dataset
from taint.detect_canary import CANARY_MARKER, CanaryDetector
from taint.detect_judge import JudgeDetector
from taint.detect_probe import ProbeDetector
from taint.harness import Harness


def _in_unit_interval(scores: list[float]) -> bool:
    return all(0.0 <= s <= 1.0 for s in scores)


def test_canary_returns_unit_scores(dataset: Dataset) -> None:
    """Canary returns one score in [0,1] per trace and counts extra runs."""
    det = CanaryDetector()
    scores = det.score_all(dataset.test)
    assert len(scores) == len(dataset.test)
    assert _in_unit_interval(scores)
    assert det.extra_runs > 0  # the canary's cost signal


def test_canary_scores_clean_traces_zero() -> None:
    """Clean traces (no injection) get score 0 and cost no extra runs.

    This is the key correctness property: the canary must not fabricate an
    injection where none existed, which would false-positive on benign traffic.
    """
    clean = [
        Harness().build_trace("sys", f"task {i}", inject=False) for i in range(8)
    ]
    det = CanaryDetector()
    scores = det.score_all(clean)
    assert all(s == 0.0 for s in scores)
    assert det.extra_runs == 0


def test_canary_uses_distinct_marker() -> None:
    """The canary's marker differs from the benchmark marker (Kerckhoffs)."""
    assert CANARY_MARKER != "PWNED"


def test_judge_returns_unit_scores(dataset: Dataset) -> None:
    """Judge returns one score in [0,1] per trace."""
    scores = JudgeDetector().score_all(dataset.test)
    assert len(scores) == len(dataset.test)
    assert _in_unit_interval(scores)


def test_judge_misses_indirect_tier2() -> None:
    """The text judge scores 0 on a tier-2 injection (no surface cue) — honest gap."""
    tier2 = Harness().build_trace("sys", "task", inject=True, tier=2)
    assert JudgeDetector().score(tier2) == 0.0


def test_probe_requires_fit(dataset: Dataset) -> None:
    """Scoring before fit raises; after fit returns unit-interval scores."""
    probe = ProbeDetector()
    try:
        probe.score_all(dataset.test)
        raise AssertionError("expected RuntimeError before fit()")
    except RuntimeError:
        pass
    probe.fit(dataset.train)
    scores = probe.score_all(dataset.test)
    assert len(scores) == len(dataset.test)
    assert _in_unit_interval(scores)


def test_detectors_do_not_read_label(dataset: Dataset) -> None:
    """A detector's score must not change if we flip the hidden label.

    Mutating ``injection_followed`` on the observable copies must not affect any
    detector's score — proving detectors score from observables only.
    """
    import copy

    test = dataset.test
    probe = ProbeDetector().fit(dataset.train)
    canary = CanaryDetector()
    judge = JudgeDetector()

    base = {
        "canary": canary.score_all(test),
        "judge": judge.score_all(test),
        "probe": probe.score_all(test),
    }

    flipped = copy.deepcopy(test)
    for t in flipped:
        t.injection_followed = not t.injection_followed

    assert CanaryDetector().score_all(flipped) == base["canary"]
    assert JudgeDetector().score_all(flipped) == base["judge"]
    assert probe.score_all(flipped) == base["probe"]
