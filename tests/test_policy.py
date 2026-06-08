"""Tests for the elevation policy + cost model."""

from __future__ import annotations

import pytest

from taint.policy import Action, CostModel, ElevationPolicy, apply_policy


def test_thresholds_map_to_actions() -> None:
    """Scores map to allow / flag / block at the configured thresholds."""
    policy = ElevationPolicy(flag_threshold=0.4, block_threshold=0.7)
    assert policy.decide(0.1) is Action.ALLOW
    assert policy.decide(0.5) is Action.FLAG
    assert policy.decide(0.9) is Action.BLOCK
    # Boundaries are inclusive at the lower edge.
    assert policy.decide(0.4) is Action.FLAG
    assert policy.decide(0.7) is Action.BLOCK


def test_invalid_thresholds_rejected() -> None:
    """Flag threshold above block threshold is rejected."""
    with pytest.raises(ValueError):
        ElevationPolicy(flag_threshold=0.8, block_threshold=0.5)


def test_policy_reduces_successful_injections() -> None:
    """Blocking high-scoring followed traces cuts successful injections.

    A perfect score (1.0 on every followed trace) under a blocking policy must
    drive residual successful injections to zero.
    """
    labels = [True, True, False, False, True, False]
    scores = [1.0, 0.9, 0.1, 0.2, 0.8, 0.0]  # high on every followed trace
    outcome = apply_policy(scores, labels, ElevationPolicy())
    assert outcome.baseline_successful == 3
    assert outcome.residual_successful == 0
    assert outcome.injections_cut_pct == 100.0
    assert outcome.false_positive_blocks == 0


def test_policy_charges_false_positives() -> None:
    """Blocking a benign trace counts as a false positive with a cost."""
    labels = [False, False, True]
    scores = [0.95, 0.1, 0.95]  # one benign trace blocked
    outcome = apply_policy(scores, labels, ElevationPolicy(), CostModel(fp_cost=2.0))
    assert outcome.false_positive_blocks == 1
    assert outcome.false_positive_rate == 0.5
    assert outcome.total_cost >= 2.0


def test_policy_prices_extra_agent_runs() -> None:
    """Extra agent runs (canary overhead) are priced into latency + cost."""
    labels = [True, False]
    scores = [0.9, 0.1]
    cost = CostModel(per_run_latency_ms=50.0)
    outcome = apply_policy(scores, labels, ElevationPolicy(), cost, extra_agent_runs=4)
    assert outcome.latency_overhead_ms == 200.0


def test_length_mismatch_raises() -> None:
    """Mismatched scores/labels lengths raise."""
    with pytest.raises(ValueError):
        apply_policy([0.5], [True, False], ElevationPolicy())
