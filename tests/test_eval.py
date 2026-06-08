"""End-to-end tests for evaluation and the CLI.

Assert that ``evaluate`` computes precision/recall/ROC-AUC for every detector,
that the canary reports overhead, that the elevation policy cuts injections, and
that the ``taint eval`` CLI runs offline and exits 0.
"""

from __future__ import annotations

from taint.cli import main
from taint.dataset import Dataset
from taint.eval import evaluate, format_table, reports_to_dict


def test_evaluate_produces_metrics_for_each_detector(dataset: Dataset) -> None:
    """Every detector gets precision, recall, F1, and ROC-AUC."""
    reports = evaluate(dataset)
    names = {r.name for r in reports}
    assert names == {"canary", "judge", "probe"}
    for r in reports:
        assert 0.0 <= r.precision <= 1.0
        assert 0.0 <= r.recall <= 1.0
        assert 0.0 <= r.f1 <= 1.0
        assert 0.0 <= r.roc_auc <= 1.0  # not NaN -> both classes present


def test_canary_reports_overhead(dataset: Dataset) -> None:
    """The canary's extra agent runs are reported as overhead; others have none."""
    reports = {r.name: r for r in evaluate(dataset)}
    assert reports["canary"].extra_agent_runs > 0
    assert reports["judge"].extra_agent_runs == 0
    assert reports["probe"].extra_agent_runs == 0


def test_policy_cuts_injections_on_benchmark(dataset: Dataset) -> None:
    """At least one detector's elevation policy cuts successful injections > 0%."""
    reports = evaluate(dataset)
    cuts = [r.policy.injections_cut_pct for r in reports]
    assert max(cuts) > 0.0
    # And the residual successful injections never exceed the baseline.
    for r in reports:
        assert r.policy.residual_successful <= r.policy.baseline_successful


def test_strongest_detector_beats_chance(dataset: Dataset) -> None:
    """The best detector clears chance-level ROC-AUC (sanity, not a leak)."""
    reports = evaluate(dataset)
    best_auc = max(r.roc_auc for r in reports)
    assert best_auc > 0.6


def test_eval_is_deterministic(dataset: Dataset) -> None:
    """Metrics (excluding wall-clock latency) are reproducible across runs."""

    def fingerprint() -> list[tuple]:
        return [
            (r.name, round(r.precision, 6), round(r.recall, 6), round(r.roc_auc, 6),
             round(r.policy.injections_cut_pct, 6), r.extra_agent_runs)
            for r in evaluate(dataset)
        ]

    assert fingerprint() == fingerprint()


def test_format_table_and_serialization(dataset: Dataset) -> None:
    """The metrics table renders and reports serialize to JSON-safe dicts."""
    reports = evaluate(dataset)
    table = format_table(reports)
    assert "detector" in table and "roc_auc" in table
    payload = reports_to_dict(reports)
    assert len(payload) == 3 and all("roc_auc" in d for d in payload)


def test_cli_eval_runs_offline(tmp_path) -> None:
    """``taint eval`` runs end-to-end offline and exits 0."""
    out = tmp_path / "metrics.json"
    code = main(["eval", "--n", "120", "--out", str(out)])
    assert code == 0
    assert out.exists() and out.stat().st_size > 0


def test_cli_gen_runs_offline(tmp_path) -> None:
    """``taint gen`` writes labeled traces offline and exits 0."""
    out = tmp_path / "traces.jsonl"
    code = main(["gen", "--n", "30", "--out", str(out)])
    assert code == 0
    assert out.exists()
