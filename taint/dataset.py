"""Build, balance, and split the labeled trace benchmark.

Turns raw :class:`~taint.harness.Trace` records into a balanced, stratified
train/test benchmark with difficulty tiers, suitable for training the probe and
scoring all detectors. Fully offline and deterministic.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from taint.harness import Harness, Trace


@dataclass
class Dataset:
    """A train/test split of labeled traces.

    Attributes:
        train: Training traces (used to fit the probe).
        test: Held-out test traces (used to score all detectors).
    """

    train: list[Trace]
    test: list[Trace]

    def labels(self, split: str = "test") -> list[bool]:
        """Return ground-truth labels for a split ("train" or "test")."""
        traces = self.train if split == "train" else self.test
        return [t.injection_followed for t in traces]

    def class_balance(self, split: str = "test") -> dict[bool, int]:
        """Return a count of each class label in the given split."""
        return dict(Counter(self.labels(split)))


def balance(traces: list[Trace], seed: int = 0) -> list[Trace]:
    """Downsample the majority class so both labels are equally represented.

    Args:
        traces: Input labeled traces.
        seed: RNG seed for deterministic selection.

    Returns:
        A class-balanced list of traces (shuffled deterministically).
    """
    rng = np.random.default_rng(seed)
    pos = [t for t in traces if t.injection_followed]
    neg = [t for t in traces if not t.injection_followed]
    k = min(len(pos), len(neg))
    if k == 0:
        # Cannot balance with a missing class; return the input unchanged.
        return list(traces)
    pos_idx = rng.choice(len(pos), size=k, replace=False)
    neg_idx = rng.choice(len(neg), size=k, replace=False)
    chosen = [pos[i] for i in pos_idx] + [neg[i] for i in neg_idx]
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def stratified_split(
    traces: list[Trace], test_frac: float = 0.4, seed: int = 0
) -> Dataset:
    """Split traces into train/test, stratified by label so both classes appear.

    Stratification guarantees both classes are present in the test set, which is
    required for ``roc_auc_score`` (it raises on a single-class set).

    Args:
        traces: Balanced labeled traces.
        test_frac: Fraction of each class assigned to the test split.
        seed: RNG seed for deterministic selection.

    Returns:
        A :class:`Dataset`.
    """
    rng = np.random.default_rng(seed)
    pos = [t for t in traces if t.injection_followed]
    neg = [t for t in traces if not t.injection_followed]

    def _split(items: list[Trace]) -> tuple[list[Trace], list[Trace]]:
        if not items:
            return [], []
        idx = rng.permutation(len(items))
        n_test = max(1, int(round(len(items) * test_frac))) if len(items) > 1 else 0
        test_idx = set(idx[:n_test].tolist())
        test = [items[i] for i in range(len(items)) if i in test_idx]
        train = [items[i] for i in range(len(items)) if i not in test_idx]
        return train, test

    pos_train, pos_test = _split(pos)
    neg_train, neg_test = _split(neg)
    return Dataset(train=pos_train + neg_train, test=pos_test + neg_test)


def build_benchmark(
    n: int = 200, test_frac: float = 0.4, seed: int = 0
) -> Dataset:
    """Generate, balance, and split a benchmark in one call.

    Args:
        n: Number of raw traces to generate.
        test_frac: Test fraction for the stratified split.
        seed: Seed for balancing/splitting.

    Returns:
        A ready-to-use :class:`Dataset`.
    """
    harness = Harness()
    raw = harness.generate(n)
    balanced = balance(raw, seed=seed)
    return stratified_split(balanced, test_frac=test_frac, seed=seed)


def save_traces(traces: list[Trace], path: str | Path) -> Path:
    """Write traces to a JSON-lines file (one observable+label record per line).

    Args:
        traces: Traces to serialize.
        path: Destination ``.jsonl`` path.

    Returns:
        The path written.
    """
    from dataclasses import asdict

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(asdict(t)) + "\n")
    return p
