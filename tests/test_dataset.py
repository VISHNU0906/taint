"""Tests for dataset balancing and the stratified train/test split."""

from __future__ import annotations

from taint.dataset import (
    Dataset,
    balance,
    build_benchmark,
    save_traces,
    stratified_split,
)
from taint.harness import Harness


def test_balance_equalizes_classes() -> None:
    """Balancing downsamples the majority class to a 50/50 split."""
    raw = Harness().generate(n=200)
    bal = balance(raw, seed=0)
    pos = sum(1 for t in bal if t.injection_followed)
    neg = sum(1 for t in bal if not t.injection_followed)
    assert pos == neg and pos > 0


def test_stratified_split_keeps_both_classes_in_test(dataset: Dataset) -> None:
    """Both labels must appear in the test split (required for ROC-AUC)."""
    bal_test = dataset.class_balance("test")
    assert bal_test.get(True, 0) > 0 and bal_test.get(False, 0) > 0
    bal_train = dataset.class_balance("train")
    assert bal_train.get(True, 0) > 0 and bal_train.get(False, 0) > 0


def test_no_trace_id_leaks_across_split(dataset: Dataset) -> None:
    """Train and test traces are disjoint (no leakage by trace id)."""
    train_ids = {t.trace_id for t in dataset.train}
    test_ids = {t.trace_id for t in dataset.test}
    assert train_ids.isdisjoint(test_ids)


def test_build_benchmark_is_deterministic() -> None:
    """The same seed yields identical train/test trace ids."""
    a = build_benchmark(n=200, seed=0)
    b = build_benchmark(n=200, seed=0)
    assert [t.trace_id for t in a.test] == [t.trace_id for t in b.test]
    assert [t.trace_id for t in a.train] == [t.trace_id for t in b.train]


def test_balance_handles_single_class() -> None:
    """Balancing a single-class input returns it unchanged (no crash)."""
    raw = [t for t in Harness().generate(n=40) if not t.injection_followed]
    assert balance(raw) == raw


def test_save_traces_writes_jsonl(tmp_path) -> None:
    """Traces serialize to one JSON record per line."""
    traces = Harness().generate(n=10)
    path = save_traces(traces, tmp_path / "traces.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
