"""White-box activation-*probe* detector (mock activations + sklearn; real TODO).

Idea
----
A linear probe (logistic regression) is trained on the model's internal
activations to find a "following-injected-instruction" direction, then used to
score new traces. Here we use the deterministic, *noisy* mock activation vectors
produced by the harness (see :func:`taint.harness._mock_activations`). Because
those vectors encode the latent behaviour only weakly and with seeded noise — and
more weakly for harder tiers — the probe is a genuine, imperfect, *generalizing*
classifier with a real train/test split, not a label leak.

# TODO(real): replace mock activations with a real residual-stream probe on an
# open model (e.g. Llama-3-8B) via Transformer hooks or `nnsight`:
#   1. Capture residual-stream activations at a chosen layer/token for each trace.
#   2. Fit this same LogisticRegression on the train split.
#   3. Score the test split. Keep the ``ProbeDetector`` interface identical so
#      eval/policy code is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from taint.harness import Trace


@dataclass
class ProbeDetector:
    """Linear activation probe over (mock) residual-stream features.

    Attributes:
        C: Inverse regularization strength for logistic regression.
        seed: Deterministic seed for the solver.
    """

    C: float = 1.0
    seed: int = 0
    _model: LogisticRegression | None = field(default=None, init=False, repr=False)
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)

    def _matrix(self, traces: list[Trace]) -> np.ndarray:
        """Stack trace activation vectors into a feature matrix."""
        return np.asarray([t.activations for t in traces], dtype=float)

    def fit(self, train: list[Trace]) -> "ProbeDetector":
        """Fit the probe on the training split's activations and labels.

        Args:
            train: Training traces (must contain both classes).

        Returns:
            self, fitted.

        Raises:
            ValueError: If the training set has only one class.
        """
        X = self._matrix(train)
        y = np.asarray([t.injection_followed for t in train], dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Probe training set must contain both classes.")
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = LogisticRegression(
            C=self.C, random_state=self.seed, max_iter=1000
        ).fit(Xs, y)
        return self

    def score(self, trace: Trace) -> float:
        """Return P(injection_followed) for a single trace."""
        return self.score_all([trace])[0]

    def score_all(self, traces: list[Trace]) -> list[float]:
        """Return P(injection_followed) for each trace in ``traces``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("ProbeDetector.score_all called before fit().")
        X = self._scaler.transform(self._matrix(traces))
        proba = self._model.predict_proba(X)[:, 1]
        return proba.astype(float).tolist()
